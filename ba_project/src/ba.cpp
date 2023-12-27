#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <ceres/rotation.h>
#include <chrono>
#include "sophus/se3.hpp"
#include <iterator>


using namespace std;
using namespace cv;
using namespace Eigen;

string img_1 = "/home/slam/slambook2/ba_project/data/1.png";
string img_2 = "/home/slam/slambook2/ba_project/data/2.png";
string img_d1 = "/home/slam/slambook2/ba_project/data/1_depth.png";
string img_d2 = "/home/slam/slambook2/ba_project/data/2_depth.png";
double fx = 520.9;
double fy = 521.0;
double cx = 325.1;
double cy = 249.7;
double camera[6] = {1,1,1,1,1,1};  //初始值


void GN_ceres(vector<Point2f> pts_2d, vector<Point3f> pts_3d, double camera[6]);
void find_feature_matches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1, std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches);
Point2d pixel2cam(const Point2d &p, const Mat &K);


struct SnavelyReprojectionError{
    SnavelyReprojectionError(double observed_x, double observed_y, Point3f point)
        : observed_x(observed_x), observed_y(observed_y), point(point){}

    template<typename T>
    bool operator()(const T* const camera,
                    T* residuals)const{
        //计算投影点
        T p[3];
        T point_w[3] = {T(point.x), T(point.y), T(point.z)};
        ceres::AngleAxisRotatePoint(camera, point_w, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T predicted_x = fx * xp + cx;
        T predicted_y = fy * yp + cy;

        residuals[0] = observed_x - predicted_x;
        residuals[1] = observed_y - predicted_y;

        return true;
    }

    double observed_x;
    double observed_y;
    Point3f point;
};

int main(int argc, char **argv)
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

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    //调用优化
    GN_ceres(pts_2d, pts_3d, camera);

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



/**
 * @brief 进行Ceres优化以解决姿态估计问题。
 *
 * 此函数使用Ceres库进行优化，解决了给定2D-3D点对的姿态估计问题。
 * 它创建了一个优化问题，将SnavelyReprojectionError应用于每个点对，
 * 并使用Ceres求解器找到最佳的相机姿态（旋转和平移）。
 * 
 * @param pts_2d 观测到的二维点的向量容器。
 * @param pts_3d 对应的三维点的向量容器
 * @param camera 用于存储结果的6元素数组。前三个元素代表旋转（以轴-角形式），
 *              后三个元素代表平移。
 * 
 * @note 此函数假设'camera'数组已经预先分配，并且有足够的空间来存储6个元素。
 *       函数结束时，'camera'将包含优化后的相机姿态参数。
 * 
 * @note 使用Eigen和Sophus库来处理3D旋转和变换。
 *       结果是旋转向量和平移向量被转换成SE(3)变换矩阵，并打印到标准输出。
 */
void GN_ceres(vector<Point2f> pts_2d, vector<Point3f> pts_3d, double camera[6]){
    // ceres优化
    ceres::Problem problem;
    for (int i = 0; i < pts_2d.size(); i++){
        double observed_x = pts_2d[i].x;
        double observed_y = pts_2d[i].y;

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6>(
                new SnavelyReprojectionError(observed_x, observed_y, pts_3d[i])),
            NULL,
            camera
        );
    }

    //配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.FullReport() << endl;

    Eigen::Vector3d rotation_vector(camera[0],camera[1],camera[2]);
    Eigen::Vector3d translation(camera[3],camera[4],camera[5]);
    // 将轴-角旋转向量转换为旋转矩阵
    Eigen::Matrix3d R = Sophus::SO3d::exp(rotation_vector).matrix();
    // 使用旋转矩阵和平移向量构建SE3变换矩阵
    Sophus::SE3d SE3_transform(R, translation);
    // 输出变换矩阵
    std::cout << "SE3 transform matrix: \n" << SE3_transform.matrix() << std::endl;
}