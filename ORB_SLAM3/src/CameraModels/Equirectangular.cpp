/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "Equirectangular.h"

#include <boost/serialization/export.hpp>

// BOOST_CLASS_EXPORT_IMPLEMENT(ORB_SLAM3::Equirectangular)

namespace ORB_SLAM3
{
    // BOOST_CLASS_EXPORT_GUID(Equirectangular, "Equirectangular")

    long unsigned int GeometricCamera::nNextId = 0;

    cv::Point2f Equirectangular::project(const cv::Point3f &p3D)
    {
        double norm = std::sqrt(p3D.x * p3D.x + p3D.y * p3D.y + p3D.z * p3D.z);
        const double lat = -std::asin(p3D.y / norm);
        const double lon = std::atan2(p3D.x / norm, p3D.z / norm);
        // convert to pixel image coordinated

        return cv::Point2f(mvParameters[0] * (0.5 + lon / (2.0 * M_PI)), mvParameters[1] * (0.5 - lat / M_PI));
    }

    Eigen::Vector2d Equirectangular::project(const Eigen::Vector3d &v3D)
    {
        Eigen::Vector2d res;
        
        const double lat = -std::asin(v3D[1] / v3D.norm());
        const double lon = std::atan2(v3D[0] / v3D.norm(), v3D[2] / v3D.norm());
        res[0] = mvParameters[0] * (0.5 + lon / (2.0 * M_PI));
        res[1] = mvParameters[1] * (0.5 - lat / M_PI);
      

        

        return res;
    }

    Eigen::Vector2f Equirectangular::project(const Eigen::Vector3f &v3D)
    {
        Eigen::Vector2f res;
        const float lat = -std::asin(v3D[1] / v3D.norm());
        const float lon = std::atan2(v3D[0] / v3D.norm(), v3D[2] / v3D.norm());
        res[0] = mvParameters[0] * (0.5 + lon / (2.0 * M_PI));
        res[1] = mvParameters[1] * (0.5 - lat / M_PI);

        return res;
    }

    Eigen::Vector2f Equirectangular::projectMat(const cv::Point3f &p3D)
    {
        cv::Point2f point = this->project(p3D);
        return Eigen::Vector2f(point.x, point.y);
    }

    float Equirectangular::uncertainty2(const Eigen::Matrix<double, 2, 1> &p2D)
    {
        return 1.0;
    }

    Eigen::Vector3f Equirectangular::unprojectEig(const cv::Point2f &p2D)
    {
        const float lon = (p2D.x / mvParameters[0] - 0.5) * (2.0 * M_PI);
        const float lat = -(p2D.y / mvParameters[1] - 0.5) * M_PI;
        // convert to equirectangular coordinates
        return Eigen::Vector3f{std::cos(lat) * std::sin(lon), -std::sin(lat), std::cos(lat) * std::cos(lon)};
    }

    cv::Point3f Equirectangular::unproject(const cv::Point2f &p2D)
    {
        const float lon = (p2D.x / mvParameters[0] - 0.5) * (2.0 * M_PI);
        const float lat = -(p2D.y / mvParameters[1] - 0.5) * M_PI;

        return cv::Point3f(std::cos(lat) * std::sin(lon), -std::sin(lat), std::cos(lat) * std::cos(lon));
    }

    Eigen::Matrix<double, 2, 3> Equirectangular::projectJac(const Eigen::Vector3d &v3D)
    {
        // const float lat = -std::asin(v3D[1]/v3D.norm());
        // const float lon = std::atan2(v3D[0], v3D[2]);
        // const float R = v3D.norm();
        Eigen::Matrix<double, 2, 3> Jac = Eigen::Matrix<double, 2, 3>::Zero();
        // Jac(0, 0) = std::cos(lon) * std::cos(lat) / R;
        // Jac(0, 1) = std::sin(lon) * std::cos(lat) / R;
        // Jac(0, 2) = -std::sin(lat) / R;
        // Jac(1, 0) = -std::sin(lon) / (R * std::sin(lat));
        // Jac(1, 1) = std::sin(lon) / (R * std::sin(lat));
        // Jac(1, 2) = 0;
        float r=v3D.norm();
        Jac(0, 0)=mvParameters[0]*v3D[2]/(2*M_PI*(v3D[0]*v3D[0]+v3D[2]*v3D[2]));
        Jac(0, 1)=0;
        Jac(0, 2)=-mvParameters[0]*v3D[0]/(2*M_PI*(v3D[0]*v3D[0]+v3D[2]*v3D[2]));
        Jac(1,0)=mvParameters[1]*v3D[0]*v3D[1]/(M_PI*r*std::sqrt(r*r-v3D[1]*v3D[1]));
        Jac(1,1)=-mvParameters[1]*std::sqrt(v3D[0]*v3D[0]+v3D[1]*v3D[1])/(M_PI*r*std::sqrt(r*r-v3D[1]*v3D[1]));
        Jac(1,2)=mvParameters[1]*v3D[1]*v3D[2]/(M_PI*r*std::sqrt(r*r-v3D[1]*v3D[1]));


        

        return Jac;
    }

    bool Equirectangular::ReconstructWithTwoViews(const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2, const std::vector<int> &vMatches12,
                                                  Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated)
    {
       
     
        std::vector<std::pair<int, int>> mvMatches12;
        std::vector<bool> mvbMatched1;
        mvMatches12.clear();
        // mvMatches12.reserve(vKeys2.size());
        mvbMatched1.resize(vKeys1.size());
        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (vMatches12[i] >= 0)
            {
                mvMatches12.push_back(std::make_pair(i, vMatches12[i]));
                mvbMatched1[i] = true;
            }
            else
                mvbMatched1[i] = false;
        }

        const int N = mvMatches12.size();
        
        Eigen::Matrix3f E_opt;
        std::vector<bool> MatchesInliers;

        if (!findEssentialMatrix_RANSAC(vKeys1, vKeys2, mvMatches12, 500, 8, E_opt, MatchesInliers))
        {
            return false;
        }

        return ReconstructE(MatchesInliers, E_opt, T21, vP3D, vbTriangulated, 1.0, 50, 1.0, vKeys1, vKeys2, mvMatches12);
    }
    bool Equirectangular::findEssentialMatrix_RANSAC(const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2, const std::vector<std::pair<int, int>> &mvMatches12, const int max_num_iter, const int min_set_size, Eigen::Matrix3f &E21, std::vector<bool> &vbMatchesInliers)
    {

        const int N = mvMatches12.size();
        // check enough parallax
        std::vector<cv::Point2f> vP1, vP2;
        for (size_t i = 0; i < N; i++)
        {

            vP1.push_back(vKeys1[mvMatches12[i].first].pt);
            vP2.push_back(vKeys2[mvMatches12[i].second].pt);
            
        }

     

        std::vector<Eigen::Vector3f> r1;
        std::vector<Eigen::Vector3f> r2;

        std::vector<bool> MatchesInliers_opt;
        std::vector<Eigen::Vector3f> min_set_r_1(min_set_size);
        std::vector<Eigen::Vector3f> min_set_r_2(min_set_size);
        int max_inliers = 0;
        for (int i = 0; i < N; i++)
        {

            r1.push_back(this->unprojectEig(vP1[i]));
            r2.push_back(this->unprojectEig(vP2[i]));

        }
        
        for (int iter = 0; iter < max_num_iter; iter++)
        {

            // 2-1. Create a minimum set
            const auto indices = createRandomIndices(N, min_set_size);

            for (unsigned int i = 0; i < min_set_size; ++i)
            {
                const auto idx = indices.at(i);
                min_set_r_1.at(i) = r1.at(idx);
                min_set_r_2.at(i) = r2.at(idx);
            }

            // 2-2. Compute the essential matrix
            Eigen::Matrix3f E;
            computeEssentialMatrix(min_set_r_1, min_set_r_2, E);
            // 2-3 Check the number of inliers
            std::vector<bool> MatchesInliers;
            int inliers_count = compute_Inliers(r1, r2, E, 1e-2, MatchesInliers);

            if (inliers_count > max_inliers)
            {
                max_inliers = inliers_count;
                E21 = E;
                
                int su = 0;

                for (int i = 0; i < N; i++)
                {
                    if (MatchesInliers.at(i))
                        su++;
                }
                vbMatchesInliers = MatchesInliers;
            }
        }
        // 3. Find the best essential matrix


        if (max_inliers < min_set_size)
        {
            return false;
        }

        return true;
    }

    void Equirectangular::computeEssentialMatrix(const std::vector<Eigen::Vector3f> &r1, const std::vector<Eigen::Vector3f> &r2, Eigen::Matrix3f &E)
    {
        const auto num_points = r1.size();

    typedef Eigen::Matrix<Eigen::Matrix3f::Scalar, Eigen::Dynamic, 9> CoeffMatrix;
    CoeffMatrix A(num_points, 9);

    for (unsigned int i = 0; i < num_points; i++) {
        A.block<1, 3>(i, 0) = r2.at(i)(0) * r1.at(i);
        A.block<1, 3>(i, 3) = r2.at(i)(1) * r1.at(i);
        A.block<1, 3>(i, 6) = r2.at(i)(2) * r1.at(i);
    }

    const Eigen::JacobiSVD<CoeffMatrix> init_svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix<Eigen::Matrix3f::Scalar, 9, 1> v = init_svd.matrixV().col(8);
    // need transpose() because elements are contained as col-major after it was constructed from a pointer
    const Eigen::Matrix3f init_E_21 = Eigen::Matrix3f(v.data()).transpose();

    const Eigen::JacobiSVD<Eigen::Matrix3f> svd(init_E_21, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix3f& U = svd.matrixU();
    Eigen::Vector3f lambda = svd.singularValues();
    const Eigen::Matrix3f& V = svd.matrixV();

    lambda(2) = 0.0;

     E = U * lambda.asDiagonal() * V.transpose();

    }

    int Equirectangular::compute_Inliers(const std::vector<Eigen::Vector3f> &r1, const std::vector<Eigen::Vector3f> &r2, const Eigen::Matrix3f &E, const float threshold, std::vector<bool> &inliers)
    {
        int num_inliers = 0;
        inliers.clear();
        for (unsigned int i = 0; i < r1.size(); ++i)
        {
            double error = r2.at(i).transpose() * E * r1.at(i);
            if (std::abs(error) < threshold)
            {
                inliers.push_back(true);
                num_inliers++;

            }
            else
            {
                
                inliers.push_back(false);
            }
        }
        return num_inliers;
    }
    std::vector<int> Equirectangular::createRandomIndices(const int N, const int min_set_size)
    {
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device()()));
        return std::vector<int>(indices.begin(), indices.begin() + min_set_size);
    }

    bool Equirectangular::ReconstructE(std::vector<bool> &vbMatchesInliers, Eigen::Matrix3f &E21, Sophus::SE3f &T21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated, float sigmaLevel, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2, const std::vector<std::pair<int, int>> &mvMatches12)
    {
        int N = 0;
        for (size_t i = 0, iend = vbMatchesInliers.size(); i < iend; i++)
            if (vbMatchesInliers[i])
                N++;

        // Compute Essential Matrix from Fundamental Matrix

        Eigen::Matrix3f R1, R2;
        Eigen::Vector3f t;

        // Recover the 4 motion hypotheses
        this->DecomposeE(E21, R1, R2, t);

        Eigen::Vector3f t1 = t;
        Eigen::Vector3f t2 = -t;

        // Reconstruct with the 4 hyphoteses and check
        std::vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
        std::vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
        float parallax1, parallax2, parallax3, parallax4;

        int nGood1 = this->CheckRT(R1, t1, vKeys1, vKeys2, mvMatches12, vbMatchesInliers, vP3D1, 4.0 * sigmaLevel * sigmaLevel, vbTriangulated1, parallax1);
        int nGood2 = this->CheckRT(R2, t1, vKeys1, vKeys2, mvMatches12, vbMatchesInliers, vP3D2, 4.0 * sigmaLevel * sigmaLevel, vbTriangulated2, parallax2);
        int nGood3 = this->CheckRT(R1, t2, vKeys1, vKeys2, mvMatches12, vbMatchesInliers, vP3D3, 4.0 * sigmaLevel * sigmaLevel, vbTriangulated3, parallax3);
        int nGood4 = this->CheckRT(R2, t2, vKeys1, vKeys2, mvMatches12, vbMatchesInliers, vP3D4, 4.0 * sigmaLevel * sigmaLevel, vbTriangulated4, parallax4);

        int maxGood = std::max(nGood1, std::max(nGood2, std::max(nGood3, nGood4)));

        int nMinGood = std::max(static_cast<int>(0.3 * N), minTriangulated);

        int nsimilar = 0;
        if (nGood1 > 0.7 * maxGood)
            nsimilar++;
        if (nGood2 > 0.7 * maxGood)
            nsimilar++;
        if (nGood3 > 0.7 * maxGood)
            nsimilar++;
        if (nGood4 > 0.7 * maxGood)
            nsimilar++;

        // If there is not a clear winner or not enough triangulated points reject initialization
        if (maxGood < nMinGood || nsimilar > 1)
        {
            
            return false;
        }
        
        // If best reconstruction has enough parallax initialize
        if (maxGood == nGood1)
        {
            if (parallax1 > minParallax)
            {
                vP3D = vP3D1;
                vbTriangulated = vbTriangulated1;

                T21 = Sophus::SE3f(R1, t1);
                return true;
            }
        }
        else if (maxGood == nGood2)
        {
            if (parallax2 > minParallax)
            {
                vP3D = vP3D2;
                vbTriangulated = vbTriangulated2;

                T21 = Sophus::SE3f(R2, t1);
                return true;
            }
        }
        else if (maxGood == nGood3)
        {
            if (parallax3 > minParallax)
            {
                vP3D = vP3D3;
                vbTriangulated = vbTriangulated3;

                T21 = Sophus::SE3f(R1, t2);
                return true;
            }
        }
        else if (maxGood == nGood4)
        {
            if (parallax4 > minParallax)
            {
                vP3D = vP3D4;
                vbTriangulated = vbTriangulated4;

                T21 = Sophus::SE3f(R2, t2);
                return true;
            }
        }

        return false;
    }

    int Equirectangular::CheckRT(const Eigen::Matrix3f &R, const Eigen::Vector3f &t, const std::vector<cv::KeyPoint> &vKeys1, const std::vector<cv::KeyPoint> &vKeys2, const std::vector<std::pair<int, int>> &vMatches12, std::vector<bool> &vbMatchesInliers, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax)
    {

        vbGood = std::vector<bool>(vKeys1.size(), false);
        vP3D.resize(vKeys1.size());

        std::vector<float> vCosParallax;
        vCosParallax.reserve(vKeys1.size());

        // Camera 1 Projection Matrix K[I|0]
        Eigen::Matrix<float, 3, 4> P1;
        P1.setZero();
        P1.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();

        Eigen::Vector3f O1;
        O1.setZero();

        // Camera 2 Projection Matrix K[R|t]
        Eigen::Matrix<float, 3, 4> P2;
        P2.block<3, 3>(0, 0) = R;
        P2.block<3, 1>(0, 3) = t;

        Eigen::Vector3f O2 = -R.transpose() * t;

        int nGood = 0;
        for (int i = 0; i < vMatches12.size(); i++)
        {

            if (!vbMatchesInliers[i])
                continue;

            const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
            const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];

            Eigen::Vector3f p3dC1;

            cv::Point3f x_p1 = this->unproject(kp1.pt);
            cv::Point3f x_p2 = this->unproject(kp2.pt);

            this->Triangulate(x_p1, x_p2, P1, P2, p3dC1);

            if (!isfinite(p3dC1(0)) || !isfinite(p3dC1(1)) || !isfinite(p3dC1(2)))
            {
                vbGood[i] = false;
                continue;
            }

            // Check parallax
            Eigen::Vector3f normal1 = p3dC1 - O1;
            float dist1 = normal1.norm();

            Eigen::Vector3f normal2 = p3dC1 - O2;
            float dist2 = normal2.norm();

            float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

            
            Eigen::Vector3f p3dC2 = R * p3dC1 + t;

            
            // Check reprojection error in first image

            Eigen::Vector2f p2D1 = project(p3dC1);

            float squareError1 = (p2D1.x() - kp1.pt.x) * (p2D1.x() - kp1.pt.x) + (p2D1.y() - kp1.pt.y) * (p2D1.y() - kp1.pt.y);

            if (squareError1 > th2)
                continue;

            // Check reprojection error in second image
            Eigen::Vector2f p2D2 = project(p3dC2);

            float squareError2 = (p2D2.x() - kp2.pt.x) * (p2D2.x() - kp2.pt.x) + (p2D2.y() - kp2.pt.y) * (p2D2.y() - kp2.pt.y);
            if (squareError2 > th2)
                continue;

            vCosParallax.push_back(cosParallax);
            vP3D[i] = cv::Point3f(p3dC1(0), p3dC1(1), p3dC1(2));
            nGood++;

            if (cosParallax < 0.9998)
                vbGood[i] = true;
        }

        if (nGood > 0)
        {
            sort(vCosParallax.begin(), vCosParallax.end());

            size_t idx = std::min(50, int(vCosParallax.size() - 1));
            parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
        }
        else
            parallax = 0;

        return nGood;
    }

    cv::Mat Equirectangular::toK()
    {
        cv::Mat K = (cv::Mat_<float>(3, 3)
                         << mvParameters[0] / (2.0 * M_PI),
                     0.f, mvParameters[0] / 2.0, 0.f, -mvParameters[1] / (M_PI), mvParameters[1] / 2.0, 0.f, 0.f, 1.0f);
        return K;
    }

    Eigen::Matrix3f Equirectangular::toK_()
    {
        Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
        K << mvParameters[0] / (2.0 * M_PI), 0.f, mvParameters[0] / 2.0, 0.f, -mvParameters[1] / (M_PI), mvParameters[1] / 2.0, 0.f, 0.f, 1.f;
        return K;
    }

    bool Equirectangular::epipolarConstrain(GeometricCamera *pCamera2, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, const float sigmaLevel, const float unc)
    {
        Eigen::Vector3f p3D;
        return this->TriangulateMatches(pCamera2, kp1, kp2, R12, t12, sigmaLevel, unc, p3D) != 0.0f;
        // return true;
    }
    bool Equirectangular::matchAndtriangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, GeometricCamera *pOther,
                                              Sophus::SE3f &Tcw1, Sophus::SE3f &Tcw2,
                                              const float sigmaLevel1, const float sigmaLevel2,
                                              Eigen::Vector3f &x3Dtriangulated)
    {
        Eigen::Matrix<float, 3, 4> eigTcw1 = Tcw1.matrix3x4();
        Eigen::Matrix3f Rcw1 = eigTcw1.block<3, 3>(0, 0);
        Eigen::Matrix3f Rwc1 = Rcw1.transpose();
        Eigen::Matrix<float, 3, 4> eigTcw2 = Tcw2.matrix3x4();
        Eigen::Matrix3f Rcw2 = eigTcw2.block<3, 3>(0, 0);
        Eigen::Matrix3f Rwc2 = Rcw2.transpose();

        cv::Point3f ray1c = this->unproject(kp1.pt);
        cv::Point3f ray2c = pOther->unproject(kp2.pt);

        Eigen::Vector3f r1(ray1c.x, ray1c.y, ray1c.z);
        Eigen::Vector3f r2(ray2c.x, ray2c.y, ray2c.z);

        // Check parallax between rays
        Eigen::Vector3f ray1 = Rwc1 * r1;
        Eigen::Vector3f ray2 = Rwc2 * r2;

        const float cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

        // If parallax is lower than 0.99, reject this match
        if (cosParallaxRays > 0.9998)
        {
            return false;
        }

        // Parallax is good, so we try to triangulate
        cv::Point3f p11, p22;

        p11.x = ray1c.x;
        p11.y = ray1c.y;
        p11.z = ray1c.z;

        p22.x = ray2c.x;
        p22.y = ray2c.y;
        p22.z = ray2c.z;

        Eigen::Vector3f x3D;

        Triangulate(p11, p22, eigTcw1, eigTcw2, x3D);

        

        // Check reprojection error in first keyframe
        //   -Transform point into camera reference system
        Eigen::Vector3f x3D1 = Rcw1 * x3D + Tcw1.translation();
        Eigen::Vector2f uv1 = this->project(x3D1);

        float errX1 = uv1(0) - kp1.pt.x;
        float errY1 = uv1(1) - kp1.pt.y;

        if ((errX1 * errX1 + errY1 * errY1) > 5.991 *sigmaLevel1)
        { // Reprojection error is high
            return false;
        }

        // Check reprojection error in second keyframe;
        //   -Transform point into camera reference system
        Eigen::Vector3f x3D2 = Rcw2 * x3D + Tcw2.translation(); // avoid using q
        Eigen::Vector2f uv2 = pOther->project(x3D2);

        float errX2 = uv2(0) - kp2.pt.x;
        float errY2 = uv2(1) - kp2.pt.y;

        if ((errX2 * errX2 + errY2 * errY2) > 5.991 *sigmaLevel2)
        { // Reprojection error is high
            return false;
        }

        // Since parallax is big enough and reprojection errors are low, this pair of points
        // can be considered as a match
        x3Dtriangulated = x3D;

        return true;
    }
    void Equirectangular::DecomposeE(const Eigen::Matrix3f &E, Eigen::Matrix3f &R1, Eigen::Matrix3f &R2, Eigen::Vector3f &t)
    {

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f Vt = svd.matrixV().transpose();

        t = U.col(2);
        t = t / t.norm();

        Eigen::Matrix3f W;
        W.setZero();
        W(0, 1) = -1;
        W(1, 0) = 1;
        W(2, 2) = 1;

        R1 = U * W * Vt;
        if (R1.determinant() < 0)
            R1 = -R1;

        R2 = U * W.transpose() * Vt;
        if (R2.determinant() < 0)
            R2 = -R2;
    }

    float Equirectangular::TriangulateMatches(GeometricCamera *pCamera2, const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3f &R12, const Eigen::Vector3f &t12, const float sigmaLevel, const float unc, Eigen::Vector3f &p3D)
    {

        Eigen::Vector3f r1 = this->unprojectEig(kp1.pt);
        Eigen::Vector3f r2 = pCamera2->unprojectEig(kp2.pt);

        // Check parallax
        Eigen::Vector3f r21 = R12 * r2;

        const float cosParallaxRays = r1.dot(r21) / (r1.norm() * r21.norm());

        if (cosParallaxRays > 0.9998)
        {
            return 0;
        }

        cv::Point3f p11, p22;

        p11.x = r1[0];
        p11.y = r1[1];
        p11.z = r1[2];

        p22.x = r2[0];
        p22.y = r2[1];
        p22.z = r2[2];

        Eigen::Vector3f x3D;
        Eigen::Matrix<float, 3, 4> Tcw1;
        Tcw1 << Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero();

        Eigen::Matrix<float, 3, 4> Tcw2;

        Eigen::Matrix3f R21 = R12.transpose();
        Tcw2 << R21, -R21 * t12;

        Triangulate(p11, p22, Tcw1, Tcw2, x3D);

        float z1 = x3D(2);
     
        float z2 = R21.row(2).dot(x3D) + Tcw2(2, 3);
        
        Eigen::Vector2f uv1 = this->project(x3D);

        float errX1 = uv1(0) - kp1.pt.x;
        float errY1 = uv1(1) - kp1.pt.y;

        if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaLevel)
        { // Reprojection error is high
            return 0;
        }

        Eigen::Vector3f x3D2 = R21 * x3D + Tcw2.col(3);
        Eigen::Vector2f uv2 = pCamera2->project(x3D2);

        float errX2 = uv2(0) - kp2.pt.x;
        float errY2 = uv2(1) - kp2.pt.y;

        if ((errX2 * errX2 + errY2 * errY2) > 5.991 * unc)
        { // Reprojection error is high
            return 0;
        }

        p3D = x3D;

        return z1;
    }
    void Equirectangular::Triangulate(const cv::Point3f &p1, const cv::Point3f &p2, const Eigen::Matrix<float, 3, 4> &Tcw1,
                                      const Eigen::Matrix<float, 3, 4> &Tcw2, Eigen::Vector3f &x3D)
    {
        
        Eigen::Matrix<float, 4, 4> A;
        A.row(0) = p1.x * Tcw1.row(2) - p1.z * Tcw1.row(0);
        A.row(1) = p1.y * Tcw1.row(2) - p1.z * Tcw1.row(1);
        A.row(2) = p2.x * Tcw2.row(2) - p2.z * Tcw2.row(0);
        A.row(3) = p2.y * Tcw2.row(2) - p2.z * Tcw2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4f x3Dh = svd.matrixV().col(3);
        x3D = x3Dh.head(3) / x3Dh(3);
    }

    std::ostream &operator<<(std::ostream &os, const Equirectangular &ph)
    {
        os << ph.mvParameters[0] << " " << ph.mvParameters[1];
        return os;
    }

    std::istream &operator>>(std::istream &is, Equirectangular &ph)
    {
        float nextParam;
        for (size_t i = 0; i < 2; i++)
        {
            assert(is.good()); // Make sure the input stream is good
            is >> nextParam;
            ph.mvParameters[i] = nextParam;
        }
        return is;
    }

    bool Equirectangular::IsEqual(GeometricCamera *pCam)
    {
        if (pCam->GetType() != GeometricCamera::CAM_EQUIRECTANGULAR)
            return false;

        Equirectangular *pEquirectangularCam = (Equirectangular *)pCam;

        if (size() != pEquirectangularCam->size())
            return false;

        bool is_same_camera = true;
        for (size_t i = 0; i < size(); ++i)
        {
            if (abs(mvParameters[i] - pEquirectangularCam->getParameter(i)) > 1e-6)
            {
                is_same_camera = false;
                break;
            }
        }
        return is_same_camera;
    }
}
