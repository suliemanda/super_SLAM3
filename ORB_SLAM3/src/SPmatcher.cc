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

#include "SPmatcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include <stdint-gcc.h>
#include "point_matching.h"
#include "read_config.h"

using namespace std;

namespace ORB_SLAM3
{

    const int SPmatcher::TH_HIGH = 100;
    const int SPmatcher::TH_LOW = 50;
    const int SPmatcher::HISTO_LENGTH = 30;

    SPmatcher::SPmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
        SuperGlueConfig superglue_config;
        superglue_config.image_width = 640;
        superglue_config.image_height = 480;
        superglue_config.dla_core = 0;
        superglue_config.input_tensor_names = {"keypoints_0", "scores_0", "descriptors_0", "keypoints_1", "scores_1", "descriptors_1"};
        superglue_config.output_tensor_names = {"scores"};
        superglue_config.onnx_file = "weights/superglue_indoor_sim_int32.onnx";
        superglue_config.onnx_file = "weights/superglue_indoor_sim_int32.engine";

        PointMatchingPtr point_matcher = std::make_shared<PointMatching>(superglue_config);
    }

    int SPmatcher::SearchByProjection(Frame &F, const vector<MapPoint *> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
    {
        // TODO: Project MapPoints
        int nmatches = 0, left = 0, right = 0;

        const bool bFactor = th != 1.0;
        std::vector<cv::KeyPoint> &kp1 = F.mvKeysUn;
        std::vector<cv::KeyPoint> kp2;
        std::vector<cv::DMatch> matches;
        std::vector<float> &scores1 = F.scores;
        std::vector<float> scores2;
        cv::Mat &descriptors1 = F.mDescriptors;
        cv::Mat descriptors2;
        Sophus::Sim3f &Scw = F.mTcw;
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
        for (size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
        {
            MapPoint *pMP = vpMapPoints[iMP];
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;
            auto uv = F.mpCamera->project(p3Dc);
            auto kp = cv::KeyPoint(uv(0), uv(1), 1);
            kp2.push_back(kp);
            descriptors2.row(iMP) = pMP->GetDescriptor();
            scores2.push_back(pMP.desc_score);
        }
        nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++)
        {
            F.mvpMapPoints[matches[i].queryIdx] = vpMapPoints[matches[i].trainIdx];
        }

        return nmatches;
    }
    // NOT USED
    float SPmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    int SPmatcher::SearchByBoW(KeyFrame *pKF, Frame &F, vector<MapPoint *> &vpMapPointMatches)
    {
        const vector<MapPoint *> vpMapPointsKF = pKF->GetMapPointMatches();

        vpMapPointMatches = vector<MapPoint *>(F.N, static_cast<MapPoint *>(NULL));
        std::vector<cv::KeyPoint> kp1;
        std::vector<cv::KeyPoint> kp2;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores1;
        std::vector<float> scores2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;
        descriptors1.create(F.mDescriptors.rows, F.mDescriptors.cols, CV_32F);
        descriptors2.create(vpMapPointsKF.size(), F.mDescriptors.cols, CV_32F);
        // descriptors2.create(pKF.mDescriptors.rows, pKF.mDescriptors.cols, CV_32F);
        for (size_t iMP = 0; iMP < F.N; iMP++)
        {
            kp1.push_back(F.mvKeys[iMP]);
            descriptors1.row(iMP) = F.mDescriptors.row(iMP);
            scores1.push_back(F.scores[iMP]);
        }
        for (size_t iMP = 0; iMP < vpMapPointsKF.size(); iMP++)
        {
            kp2.push_back(vpMapPointsKF[iMP]->GetKeyPoint());
            descriptors2.row(iMP) = vpMapPointsKF[iMP]->GetDescriptor();
            scores2.push_back(vpMapPointsKF[iMP].desc_score);
        }
        int nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++)
        {
            MapPoint *pMP = vpMapPointsKF[matches[i].trainIdx];
            // F.mvpMapPoints[matches[i].queryIdx] = pMP;
            vpMapPointMatches[matches[i].queryIdx] = pMP;
        }

        return nmatches;
    }

    int SPmatcher::SearchByProjection(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints,
                                      vector<MapPoint *> &vpMatched, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        // const float &fx = pKF->fx;
        // const float &fy = pKF->fy;
        // const float &cx = pKF->cx;
        // const float &cy = pKF->cy;

        std::vector<cv::KeyPoint> kp1;
        std::vector<cv::KeyPoint> kp2;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores1;
        std::vector<float> scores2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;
        descriptors1.create(pKF->mDescriptors.rows, pKF->mDescriptors.cols, CV_32F);
        descriptors2.create(vpPoints.size(), pKF->mDescriptors.cols, CV_32F);
        for (size_t iMP = 0; iMP < pKF->N; iMP++)
        {
            kp1.push_back(pKF->mvKeys[iMP]);
            descriptors1.row(iMP) = pKF->mDescriptors.row(iMP);
            scores1.push_back(pKF->scores[iMP]);
        }

        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

        int nmatches = 0;

        // For each Candidate MapPoint Project and Match
        // TODO: if camera is spherical all points are in image
        for (int iMP = 0, iendMP = vpPoints.size(); iMP < iendMP; iMP++)
        {
            MapPoint *pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if (pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            // if (p3Dc(2) < 0.0)
            // continue;

            // Project into Image
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);
            auto kp = cv::KeyPoint(uv(0), uv(1), 1);
            kp2.push_back(kp);
            descriptors2.row(iMP) = pMP->GetDescriptor();
            scores2.push_back(pMP.desc_score);
        }
        nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++)
        {

            vpMatched[matches[i].queryIdx] = vpPoints[matches[i].trainIdx];
        }

        return nmatches;
    }

    int SPmatcher::SearchByProjection(KeyFrame *pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint *> &vpPoints, const std::vector<KeyFrame *> &vpPointsKFs,
                                      std::vector<MapPoint *> &vpMatched, std::vector<KeyFrame *> &vpMatchedKF, int th, float ratioHamming)
    {
        // Get Calibration Parameters for later projection
        // const float &fx = pKF->fx;
        // const float &fy = pKF->fy;
        // const float &cx = pKF->cx;
        // const float &cy = pKF->cy;
        //

        std::vector<cv::KeyPoint> kp1;
        std::vector<cv::KeyPoint> kp2;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores1;
        std::vector<float> scores2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
        // Eigen::Vector3f Ow = Tcw.inverse().translation();
        descriptors1.create(pKF->mDescriptors.rows, pKF->mDescriptors.cols, CV_32F);
        descriptors2.create(vpPoints.size(), pKF->mDescriptors.cols, CV_32F);

        // Set of MapPoints already found in the KeyFrame
        set<MapPoint *> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint *>(NULL));

        int nmatches = 0;
        for (size_t iMP = 0; iMP < pKF->N; iMP++)
        {
            kp1.push_back(pKF->mvKeys[iMP]);
            descriptors1.row(iMP) = pKF->mDescriptors.row(iMP);
            scores1.push_back(pKF->scores[iMP]);
        }
        for (size_t iMP = 0; iMP < vpPoints.size(); iMP++)
        {
            if (vpPoints[iMP]->isBad() || spAlreadyFound.count(vpPoints[iMP]))
                continue;
            Eigen::Vector3f p3Dw = vpPoints[iMP]->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;
            auto uv = pKF->mpCamera->project(p3Dc);
            auto kp = cv::KeyPoint(uv(0), uv(1), 1);
            kp2.push_back(kp);
            descriptors2.row(iMP) = vpPoints[iMP]->GetDescriptor();
            scores2.push_back(vpPoints[iMP].desc_score);
        }
        nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++)
        {
            vpMatched[matches[i].queryIdx] = vpPoints[matches[i].trainIdx];
            vpMatchedKF[matches[i].queryIdx] = vpPointsKFs[matches[i].trainIdx];
        }

        return nmatches;
    }

    int SPmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    {
        int nmatches = 0;
        vnMatches12 = vector<int>(F1.mvKeysUn.size(), -1);
        std::vector<cv::KeyPoint> &kp1 = F1.mvKeysUn;
        std::vector<cv::KeyPoint> &kp2 = F2.mvKeysUn;
        std::vector<cv::DMatch> matches;
        std::vector<float> &scores1 = F1.scores;
        std::vector<float> &scores2 = F2.scores;
        cv::Mat &descriptors1 = F1.mDescriptors;
        cv::Mat &descriptors2 = F2.mDescriptors;
        // vector<int> vnMatches21(F2.mvKeysUn.size(), -1);
        nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++)
        {
            vnMatches12[matches[i].queryIdx] = matches[i].trainIdx;
            // vnMatches21[matches[i].trainIdx] = matches[i].queryIdx;
        }
        for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
            if (vnMatches12[i1] >= 0)
                vbPrevMatched[i1] = F2.mvKeysUn[vnMatches12[i1]].pt;

        // descriptors1.create(F1.mDescriptors.rows, F1.mDescriptors.cols, CV_32F);
        // descriptors2.create(F2.mDescriptors.rows, F2.mDescriptors.cols, CV_32F);

        // Update prev matched

        return nmatches;
    }

    int SPmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        std::vector<cv::KeyPoint> kp1;
        std::vector<cv::KeyPoint> kp2;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores1;
        std::vector<float> scores2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;
        vpMatches12 = vector<MapPoint *>(vpMapPoints1.size(), static_cast<MapPoint *>(NULL));

        descriptors1.create(pKF1->mDescriptors.rows, pKF1->mDescriptors.cols, CV_32F);
        descriptors2.create(pKF2->mDescriptors.rows, pKF2->mDescriptors.cols, CV_32F);
        kp1 = pKF1->mvKeysUn;
        kp2 = pKF2->mvKeysUn;
        descriptors1 = pKF1->mDescriptors;
        descriptors2 = pKF2->mDescriptors;
        scores1 = pKF1->scores;
        scores2 = pKF2->scores;

        // for (size_t iMP = 0; iMP < pKF1->N; iMP++)
        // {
        // kp1.push_back(pKF1->mvKeysUn[iMP]);
        // descriptors1.row(iMP) = pKF1->mDescriptors.row(iMP);
        // scores1.push_back(pKF1->scores[iMP]);
        // }
        // for (size_t iMP = 0; iMP < pKF2->N; iMP++)
        // {
        // kp2.push_back(pKF2->mvKeysUn[iMP]);
        // descriptors2.row(iMP) = pKF2->mDescriptors.row(iMP);
        // scores2.push_back(pKF2->scores[iMP]);
        // }
        int nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++)
        {
            vpMatches12[matches[i].queryIdx] = pKF2->GetMapPointMatches()[matches[i].trainIdx];
        }

        return nmatches;
    }

    int SPmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                          vector<pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        // const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        // const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        std::vector<cv::KeyPoint> &kps1 = pKF1->mvKeysUn;
        std::vector<cv::KeyPoint> &kps2 = pKF2->mvKeysUn;
        std::vector<cv::DMatch> matches;
        std::vector<float> &scores1 = pKF1->scores;
        std::vector<float> &scores2 = pKF2->scores;
        cv::Mat &descriptors1 = pKF1->mDescriptors;
        cv::Mat &descriptors2 = pKF2->mDescriptors;

        // descriptors1.create(pKF1->mDescriptors.rows, pKF1->mDescriptors.cols, CV_32F);
        // descriptors2.create(pKF2->mDescriptors.rows, pKF2->mDescriptors.cols, CV_32F);

        // Compute epipole in second image
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = pKF1->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;

        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        GeometricCamera *pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

        if (!pKF1->mpCamera2 && !pKF2->mpCamera2)
        {
            T12 = T1w * Tw2;
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
        }
        else
        {
            Sophus::SE3f Tr1w = pKF1->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr = Tlr.rotationMatrix(), Rrl = Trl.rotationMatrix(), Rrr = Trr.rotationMatrix();
        Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        int nmatches = 0;
        nmatches = this->SuperGluematcher(kps1, kps2, descriptors1, descriptors2, scores1, scores2, matches);
        vector<bool> vbMatched2(pKF2->N, false);
        vector<int> vMatches12(pKF1->N, -1);
        for (int i = 0; i < nmatches; i++)
        {
            vMatches12[matches[i].queryIdx] = matches[i].trainIdx;
            vbMatched2[matches[i].trainIdx] = true;
        }

        for (size_t i = 0, iend = vMatches12.size(); i < iend; i++)
        {
            if (vMatches12[i] < 0)
                continue;
            vMatchedPairs.push_back(make_pair(i, vMatches12[i]));
        }
        // TODO: check epipolar  constraints

        return nmatches;
    }

    int SPmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        GeometricCamera *pCamera;
        Sophus::SE3f Tcw;
        Eigen::Vector3f Ow;
        std::vector<cv::KeyPoint> &kps1 = pKF->mvKeysUn;
        std::vector<cv::KeyPoint> kps2;
        std::vector<cv::DMatch> matches;
        std::vector<float> &scores1 = pKF->scores;
        std::vector<float> scores2;
        cv::Mat &descriptors1 = pKF->mDescriptors;
        cv::Mat descriptors2;
        descriptors2.create(vpMapPoints.size(), pKF->mDescriptors.cols, CV_32F);

        if (bRight)
        {
            Tcw = pKF->GetRightPose();
            Ow = pKF->GetRightCameraCenter();
            pCamera = pKF->mpCamera2;
        }
        else
        {
            Tcw = pKF->GetPose();
            Ow = pKF->GetCameraCenter();
            pCamera = pKF->mpCamera;
        }

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused = 0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad = 0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal = 0, count_notidx = 0, count_thcheck = 0;
        for (int i = 0; i < nMPs; i++)
        {
            MapPoint *pMP = vpMapPoints[i];

            if (!pMP)
            {
                count_notMP++;
                continue;
            }

            if (pMP->isBad())
            {
                count_bad++;
                continue;
            }
            else if (pMP->IsInKeyFrame(pKF))
            {
                count_isinKF++;
                continue;
            }

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if (p3Dc(2) < 0.0f)
            {
                count_negdepth++;
                continue;
            }

            const float invz = 1 / p3Dc(2);

            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            // Point must be inside the image
            if (pCamera->GetType() != 2)
            {
                if (!pKF->IsInImage(uv(0), uv(1)))
                {
                    count_notinim++;
                    continue;
                }
            }
            const float ur = uv(0) - bf * invz;

            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            // Eigen::Vector3f PO = p3Dw - Ow;
            // const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            // if (dist3D < minDistance || dist3D > maxDistance)
            // {
            // count_dist++;
            // continue;
            // }

            // Viewing angle must be less than 60 deg
            if (pCamera->getType() != 2)
            {
                Eigen::Vector3f Pn = pMP->GetNormal();

                if (PO.dot(Pn) < 0.5 * dist3D)
                {
                    count_normal++;
                    continue;
                }
            }
            // int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            // const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            // const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), radius, bRight);

            // if (vIndices.empty())
            // {
            // count_notidx++;
            // continue;
            // }

            // Match to the most similar keypoint in the radius

            // const cv::Mat dMP = pMP->GetDescriptor();
            descriptors2.row(i) = pMP->GetDescriptor();
            kps2.push_back(cv::KeyPoint(uv(0), uv(1), 1));
            scores2.push_back(pMP.desc_score);
        }
        int nmatches = this->SuperGluematcher(kps1, kps2, descriptors1, descriptors2, scores1, scores2, matches);

        for (int i = 0; i < nmatches; i++)
        {
            MapPoint *pMP = vpMapPoints[matches[i].trainIdx];
            // if (pMP->isBad())
            // continue;
            // if (pMP->IsInKeyFrame(pKF))
            // continue;
            // pMP->AddObservation(pKF, matches[i].trainIdx);
            // pKF->AddMapPoint(pMP, matches[i].queryIdx);
            // nFused++;

            // const cv::KeyPoint &kp = (pKF->NLeft == -1) ? pKF->mvKeysUn[idx]
            //  : (!bRight)        ? pKF->mvKeys[idx]
            // : pKF->mvKeysRight[idx];
            //
            // const int &kpLevel = kp.octave;

            // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
            // continue;
            const cv::KeyPoint &kp = kps1[matches[i].queryIdx];
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;
            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            const float &kpx = kp.pt.x;
            const float &kpy = kp.pt.y;
            const float ex = uv(0) - kpx;
            const float ey = uv(1) - kpy;
            const float e2 = ex * ex + ey * ey;

            if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                continue;

            MapPoint *pMPinKF = pKF->GetMapPoint(matches[i].queryIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                {
                    if (pMPinKF->Observations() > pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF, matches[i].queryIdx);
                pKF->AddMapPoint(pMP, matches[i].queryIdx);
            }
            nFused++;
        }

        // If there is already a MapPoint replace otherwise add new measurement

        return nFused;
    }

    int SPmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(), Scw.translation() / Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        const set<MapPoint *> spAlreadyFound = pKF->GetMapPoints();

        int nFused = 0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for (int i = 0; i < nMPs; i++)
        {
            MapPoint *pMP = vpPoints[i];

            if (!pMP)
            {

                continue;
            }

            if (pMP->isBad())
            {

                continue;
            }
            else if (pMP->IsInKeyFrame(pKF))
            {

                continue;
            }

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if (p3Dc(2) < 0.0f)
            {

                continue;
            }

            const float invz = 1 / p3Dc(2);

            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            // Point must be inside the image
            if (pCamera->GetType() != 2)
            {
                if (!pKF->IsInImage(uv(0), uv(1)))
                {

                    continue;
                }
            }

            // const float maxDistance = pMP->GetMaxDistanceInvariance();
            // const float minDistance = pMP->GetMinDistanceInvariance();
            // Eigen::Vector3f PO = p3Dw - Ow;
            // const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            // if (dist3D < minDistance || dist3D > maxDistance)
            // {
            // count_dist++;
            // continue;
            // }

            // Viewing angle must be less than 60 deg
            if (pCamera->getType() != 2)
            {
                Eigen::Vector3f Pn = pMP->GetNormal();

                if (PO.dot(Pn) < 0.5 * dist3D)
                {
                    continue;
                }
            }
            // int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

            // Search in a radius
            // const float radius = th * pKF->mvScaleFactors[nPredictedLevel];

            // const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0), uv(1), radius, bRight);

            // if (vIndices.empty())
            // {
            // count_notidx++;
            // continue;
            // }

            // Match to the most similar keypoint in the radius

            // const cv::Mat dMP = pMP->GetDescriptor();
            descriptors2.row(i) = pMP->GetDescriptor();
            kps2.push_back(cv::KeyPoint(uv(0), uv(1), 1));
            scores2.push_back(pMP.desc_score);
        }
        int nmatches = this->SuperGluematcher(kps1, kps2, descriptors1, descriptors2, scores1, scores2, matches);

        for (int i = 0; i < nmatches; i++)
        {
            MapPoint *pMP = vpPoints[matches[i].trainIdx];
            // if (pMP->isBad())
            // continue;
            // if (pMP->IsInKeyFrame(pKF))
            // continue;
            // pMP->AddObservation(pKF, matches[i].trainIdx);
            // pKF->AddMapPoint(pMP, matches[i].queryIdx);
            // nFused++;

            // const cv::KeyPoint &kp = (pKF->NLeft == -1) ? pKF->mvKeysUn[idx]
            //  : (!bRight)        ? pKF->mvKeys[idx]
            // : pKF->mvKeysRight[idx];
            //
            // const int &kpLevel = kp.octave;

            // if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
            // continue;
            const cv::KeyPoint &kp = kps1[matches[i].queryIdx];
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;
            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            const float &kpx = kp.pt.x;
            const float &kpy = kp.pt.y;
            const float ex = uv(0) - kpx;
            const float ey = uv(1) - kpy;
            const float e2 = ex * ex + ey * ey;

            if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                continue;

            // If there is already a MapPoint replace otherwise add new measurement

            MapPoint *pMPinKF = pKF->GetMapPoint(matches[i].queryIdx);
            if (pMPinKF)
            {
                if (!pMPinKF->isBad())
                    vpReplacePoint[matches[i].trainIdx] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF, matches[i].queryIdx);
                pKF->AddMapPoint(pMP, matches[i].queryIdx);
            }
            nFused++;
        }

        return nFused;
    }

    // NOT USED
    int SPmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
    {
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 & 2 from world
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();

        // Transformation between cameras
        Sophus::Sim3f S21 = S12.inverse();

        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
        const int N1 = vpMapPoints1.size();

        const vector<MapPoint *> vpMapPoints2 = pKF2->GetMapPointMatches();
        const int N2 = vpMapPoints2.size();

        vector<bool> vbAlreadyMatched1(N1, false);
        vector<bool> vbAlreadyMatched2(N2, false);

        for (int i = 0; i < N1; i++)
        {
            MapPoint *pMP = vpMatches12[i];
            if (pMP)
            {
                vbAlreadyMatched1[i] = true;
                int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
                if (idx2 >= 0 && idx2 < N2)
                    vbAlreadyMatched2[idx2] = true;
            }
        }

        vector<int> vnMatch1(N1, -1);
        vector<int> vnMatch2(N2, -1);

        // Transform from KF1 to KF2 and search
        for (int i1 = 0; i1 < N1; i1++)
        {
            MapPoint *pMP = vpMapPoints1[i1];

            if (!pMP || vbAlreadyMatched1[i1])
                continue;

            if (pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc1 = T1w * p3Dw;
            Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

            // Depth must be positive
            if (p3Dc2(2) < 0.0)
                continue;

            const float invz = 1.0 / p3Dc2(2);
            const float x = p3Dc2(0) * invz;
            const float y = p3Dc2(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (!pKF2->IsInImage(u, v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc2.norm();

            // Depth must be inside the scale invariance region
            if (dist3D < minDistance || dist3D > maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

            // Search in a radius
            const float radius = th * pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

                if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_HIGH)
            {
                vnMatch1[i1] = bestIdx;
            }
        }

        // Transform from KF2 to KF2 and search
        for (int i2 = 0; i2 < N2; i2++)
        {
            MapPoint *pMP = vpMapPoints2[i2];

            if (!pMP || vbAlreadyMatched2[i2])
                continue;

            if (pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc2 = T2w * p3Dw;
            Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

            // Depth must be positive
            if (p3Dc1(2) < 0.0)
                continue;

            const float invz = 1.0 / p3Dc1(2);
            const float x = p3Dc1(0) * invz;
            const float y = p3Dc1(1) * invz;

            const float u = fx * x + cx;
            const float v = fy * y + cy;

            // Point must be inside the image
            if (!pKF1->IsInImage(u, v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc1.norm();

            // Depth must be inside the scale pyramid of the image
            if (dist3D < minDistance || dist3D > maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

            // Search in a radius of 2.5*sigma(ScaleLevel)
            const float radius = th * pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

            if (vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

                if (kp.octave < nPredictedLevel - 1 || kp.octave > nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_HIGH)
            {
                vnMatch2[i2] = bestIdx;
            }
        }

        // Check agreement
        int nFound = 0;

        for (int i1 = 0; i1 < N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if (idx2 >= 0)
            {
                int idx1 = vnMatch2[idx2];
                if (idx1 == i1)
                {
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    int SPmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0;
        std::vector<cv::KeyPoint> kps1;
        std::vector<cv::KeyPoint> kps2;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores1;
        std::vector<float> scores2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;

        // Rotation Histogram (to check rotation consistency)

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        const Eigen::Vector3f twc = Tcw.inverse().translation();

        const Sophus::SE3f Tlw = LastFrame.GetPose();
        const Eigen::Vector3f tlc = Tlw * twc;

        for (int i = 0; i < LastFrame.N; i++)
        {
            MapPoint *pMP = LastFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (!LastFrame.mvbOutlier[i])
                {
                    // Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0 / x3Dc(2);

                    if (invzc < 0)
                        continue;

                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                        continue;
                    if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                        continue;
                    kp2.push_back(cv::KeyPoint(uv(0), uv(1), 1));
                    scores2.push_back(pMP.desc_score);
                    descriptors2.push_back(pMP->GetDescriptor());
                }
            }
        }
        for (int i = 0; i < CurrentFrame.N; i++)
        {
            if (CurrentFrame.mvpMapPoints[i])
                if (CurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    continue;
            kp1.push_back(CurrentFrame.mvKeysUn[i]);
            scores1.push_back(CurrentFrame.scores[i]);
            descriptors1.push_back(CurrentFrame.mDescriptors.row(i));
        }
        nmatches = this->SuperGluematcher(kp1, kp2, descriptors1, descriptors2, scores1, scores2, matches);

        for (int i = 0; i < nmatches; i++)
        {

            MapPoint *pMP = LastFrame.mvpMapPoints[matches[i].queryIdx];
            // check projection error
            const cv::KeyPoint &kp = CurrentFrame.mvKeysUn[matches[i].trainIdx];
            Eigen::Vector3f x3Dw = pMP->GetWorldPos();
            Eigen::Vector3f x3Dc = Tcw * x3Dw;
            auto uv = CurrentFrame.mpCamera->project(x3Dc);
            const float &kpx = kp.pt.x;
            const float &kpy = kp.pt.y;
            const float ex = uv(0) - kpx;
            const float ey = uv(1) - kpy;
            const float e2 = ex * ex + ey * ey;
            if (e2 * CurrentFrame.mvInvLevelSigma2[kp.octave] > 5.99)
            {
                nmatches--;
                continue;
            }
            CurrentFrame.mvpMapPoints[i] = pMP;
        }

        return nmatches;
    }

    int SPmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint *> &sAlreadyFound, const float th, const int ORBdist)
    {
        int nmatches = 0;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        Eigen::Vector3f Ow = Tcw.inverse().translation();
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();
        std::vector<cv::KeyPoint> kps1;
        std::vector<cv::KeyPoint> kps2;
        std::vector<cv::DMatch> matches;
        std::vector<float> scores1;
        std::vector<float> scores2;
        cv::Mat descriptors1;
        cv::Mat descriptors2;

        for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMPs[i];

            if (pMP)
            {
                if (!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    // Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if (uv(0) < CurrentFrame.mnMinX || uv(0) > CurrentFrame.mnMaxX)
                        continue;
                    if (uv(1) < CurrentFrame.mnMinY || uv(1) > CurrentFrame.mnMaxY)
                        continue;

                    kps2.push_back(cv::KeyPoint(uv(0), uv(1), 1));
                    scores2.push_back(pMP.desc_score);
                    descriptors2.push_back(pMP->GetDescriptor());
                }
            }
        }
        for (int i = 0; i < CurrentFrame.N; i++){
            if (CurrentFrame.mvpMapPoints[i])
                            continue;
            kps1.push_back(CurrentFrame.mvKeysUn[i]);
            scores1.push_back(CurrentFrame.scores[i]);
            descriptors1.push_back(CurrentFrame.mDescriptors.row(i));
        }
        nmatches = this->SuperGluematcher(kps1, kps2, descriptors1, descriptors2, scores1, scores2, matches);
        for (int i = 0; i < nmatches; i++){
            MapPoint *pMP = vpMPs[matches[i].trainIdx];
            CurrentFrame.mvpMapPoints[matches[i].queryIdx] = pMP;


        }

                   

        return nmatches;
    }

    // NOT USED
    void SPmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();
            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }

    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    // NOT USED
    float SPmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        float dist = (float)cv::norm(a, b, cv::NORM_L2);

        return dist;
    }
    int SPmatcher::SuperGluematcher(std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2, cv::Mat &des1, cv::Mat &des2, std::vector<float> &scores1,
                                    std::vector<float> &scores2, std::vector<cv::DMatch> &matches)
    {
        Eigen::Matrix<double, 259, Eigen::Dynamic> &features0;
        Eigen::Matrix<double, 259, Eigen::Dynamic> &features1;
        feature0.resize(259, kp1.size());
        feature1.resize(259, kp2.size());
        for (int i = 0; i < kp1.size(); i++)
        {
            feature0(0, i) = kp1[i].pt.x;
            feature0(1, i) = kp1[i].pt.y;
            feature0(2, i) = scores1[i];
            for (int j = 0; j < 256; j++)
            {
                feature0(j + 3, i) = des1.at<uchar>(i, j);
            }
        }
        for (int i = 0; i < kp2.size(); i++)
        {
            feature1(0, i) = kp2[i].pt.x;
            feature1(1, i) = kp2[i].pt.y;
            feature1(2, i) = scores2[i];
            for (int j = 0; j < 256; j++)
            {
                feature1(j + 3, i) = des2.at<uchar>(i, j);
            }
        }
        point_matcher->MatchingPoints(feature0, feature1, matches, false);
        return matches.size();
    }

} // namespace ORB_SLAM
