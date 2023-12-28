#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace g2o;

string img_1 = "../data/1.png";
string img_2 = "../data/2.png";
string img_d1 = "../data/1_depth.png";
string img_d2 = "../data/2_depth.png";
double fx = 520.9;
double fy = 521.0;
double cx = 325.1;
double cy = 249.7;
double camera[6] = {1,1,1,1,1,1};  //初始值

Point2d pixel2cam(const Point2d &p, const Mat &K);

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches);

void BA_g2o(const vector<Point3f> &pts_3d,
            const vector<Point2f> &pts_2d);

// BA的顶点 参数模板：优化变量纬度和数据类型 x
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    //重置
    virtual void setToOriginImpl() override{
        _estimate = Sophus::SE3d(); //_estimate是模板参数
    }
    //更新
    virtual void oplusImpl(const double* update) override{
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }
    //读取和写入
    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override{}
};

//BA的边 模板参数：观测值纬度，类型，连接点类型 e
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjection(const Eigen::Vector3d &point) : _point(point){}

    virtual void computeError() override{
        const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]); // 多态：基类指针(_vertices[0])可以指向派生类对象(VertexPose)
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d p_c = T * _point;
        _error = _measurement - Eigen::Vector2d(fx * p_c[0] / p_c[2] + cx, fy * p_c[1] / p_c[2] + cy);
    }   

    //雅可比(不写会根据computeError自己算)
    // virtual void linearizeOplus() override{
    //     const VertexPose *v = static_cast<const VertexPose *>(_vertices[0]);
    //     Sophus::SE3d T = v->estimate();
    //     Eigen::Vector3d p_c = T * _point;
    //     double X = p_c[0];
    //     double Y = p_c[1];
    //     double Z = p_c[2];
    //     _jacobianOplusXi
    //         << -fx / Z,
    //         0, fx * X / (Z * Z), fx * X * Y / (Z * Z), -fx - fx * X * X / (Z * Z), fx * Y / Z,
    //         0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / (Z * Z), -fy * X * Y / (Z * Z), -fy * X / Z;
    // }

    //读取和写入
    virtual bool read(istream &in) override{}
    virtual bool write(ostream &out) const override{}

private:
    Eigen::Vector3d _point;
};

int main(int argc, char const **argv)
{
    Mat img1 = imread(img_1,CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(img_2,CV_LOAD_IMAGE_COLOR);
    if(img1.empty() && img2.empty()){
        cerr << "请输入正确的路径" << endl;
        return -1;
    }

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    //orb匹配
    find_feature_matches(img1, img2, keypoints_1, keypoints_2, matches);

    //深度图
    Mat d1 = imread(img_d1, CV_LOAD_IMAGE_UNCHANGED);
        if(d1.empty()){
        cerr << "请输入正确的路径" << endl;
        return -1;
    }
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for(DMatch m : matches){
        ushort d = d1.at<unsigned short>(int(keypoints_1[m.queryIdx].pt.y), int(keypoints_1[m.queryIdx].pt.x));
        if(d == 0) continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(Point2f(keypoints_2[m.trainIdx].pt));
    }

    BA_g2o(pts_3d, pts_2d);

    cout << "3d-2d pairs: " << pts_3d.size() << endl;
    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches){

    Mat descriptions1, descriptions2;
    auto detector = ORB::create();
    auto descriptor = ORB::create();
    auto matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //检测角点
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    //计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptions1);
    descriptor->compute(img_2, keypoints_2, descriptions2);
    //匹配
    vector<DMatch> match;
    matcher->match(descriptions1, descriptions2, match);

    //挑选好的匹配 找到最近和最远的汉明距离
    double mind = 10000;
    double maxd = 0;
    for (int i = 0; i < match.size(); i++){
        double dist = match[i].distance;
        if(dist < mind) mind = dist;
        if(dist > maxd) maxd = dist;
    }
    printf("-- Max dist : %f \n", maxd);
    printf("-- Min dist : %f \n", mind);

    for (int i = 0; i < descriptions1.rows; i++){
        if(match[i].distance <= max(2*mind , 30.0)){
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}


void BA_g2o(const vector<Point3f> &pts_3d,
            const vector<Point2f> &pts_2d
            ){
    //每个误差
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; //每个误差项优化变量纬度为6，误差项纬度为3(齐次坐标)
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; //线性求解器类型
    //梯度下降
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer; //图模式
    optimizer.setAlgorithm(solver); //设置求解器
    optimizer.setVerbose(true); //打开调试输出

    //设置顶点
    VertexPose *vertex = new VertexPose();
    vertex->setId(0);
    vertex->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex);

    //添加边
    for (size_t i = 0; i < pts_3d.size(); i++){
        EdgeProjection *edge = new EdgeProjection(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        edge->setId(i);
        edge->setVertex(0, vertex);
        edge->setMeasurement(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    Sophus::SE3d optimized_pose = vertex->estimate();
    cout << "\n"
         << "T="
         << "\n"
         << optimized_pose.matrix() << endl;
}