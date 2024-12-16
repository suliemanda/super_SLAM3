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

/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "SPextractor.h"
#include "super_point.h"
#include "read_config.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

    const float factorPI = (float)(CV_PI / 180.f);

    SPextractor::SPextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                             int _iniThFAST, int _minThFAST) : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
                                                               iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        SuperPointConfig super_point_config;
        super_point_config.keypoint_threshold = _iniThFAST * 1e-3;
        super_point_config.engine_file = "weights/superpoint_v1_sim_int32.engine";
        super_point_config.onnx_file = "weights/superpoint_v1_sim_int32.onnx";
        super_point_config.max_keypoints = _nfeatures;
        super_point_config.input_tensor_names = {"input"};
        super_point_config.output_tensor_names = {"scores", "descriptors"};
        super_point_config.dla_core = -1;
        super_point_config.remove_borders = 20;
        SPmodel = std::make_shared<SuperPoint>(SuperPoint(super_point_config));
        if (!SPmodel->build())
        {
            std::cout << "Error in SuperPoint building" << std::endl;
            exit(0);
        }

        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;
        for (int i = 1; i < nlevels; i++)
        {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for (int i = 0; i < nlevels; i++)
        {
            mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
            mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale =
            nfeatures * (1 - factor) /
            (1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for (int level = 0; level < nlevels - 1; level++)
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x - UL.x) / 2);
        const int halfY = ceil(static_cast<float>(BR.y - UL.y) / 2);

        // Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x + halfX, UL.y);
        n1.BL = cv::Point2i(UL.x, UL.y + halfY);
        n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x, UL.y + halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x, BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        // Associate points to childs
        for (size_t i = 0; i < vKeys.size(); i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if (kp.pt.x < n1.UR.x)
            {
                if (kp.pt.y < n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if (kp.pt.y < n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if (n1.vKeys.size() == 1)
            n1.bNoMore = true;
        if (n2.vKeys.size() == 1)
            n2.bNoMore = true;
        if (n3.vKeys.size() == 1)
            n3.bNoMore = true;
        if (n4.vKeys.size() == 1)
            n4.bNoMore = true;
    }

    static bool compareNodes(pair<int, ExtractorNode *> &e1, pair<int, ExtractorNode *> &e2)
    {
        if (e1.first < e2.first)
        {
            return true;
        }
        else if (e1.first > e2.first)
        {
            return false;
        }
        else
        {
            if (e1.second->UL.x < e2.second->UL.x)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    int SPextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint> &_keypoints,
                                OutputArray _descriptors,std::vector<float> &scores, std::vector<int> &vLappingArea)
    {
        if (_image.empty())
            return -1;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);
        Mat mask = _mask.getMat();
        vector<KeyPoint> keypoints;
        SPmodel->infer(image, mask, keypoints, _descriptors,scores);
        // SPmodel->visualization("superpoint", image);
        Mat descriptors;
        int nkeypoints = 0;
        nkeypoints = (int)keypoints.size();

        if (nkeypoints == 0)
            _descriptors.release();
        else
        {
            descriptors = _descriptors.getMat();
        }
        _keypoints.clear();
        _keypoints.reserve(nkeypoints);
        _keypoints = vector<cv::KeyPoint>(nkeypoints, KeyPoint());

        // Modified for speeding up stereo fisheye matching
        int monoIndex = 0, stereoIndex = nkeypoints - 1;
        int nkeypointsLevel = (int)keypoints.size();
        Mat desc;
        descriptors.copyTo(desc);
        float scale = mvScaleFactor[0];
        int i = 0;
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                        keypointEnd = keypoints.end();
             keypoint != keypointEnd; ++keypoint)
        {

            if (keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1])
            {

                _keypoints.at(stereoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(stereoIndex));
                stereoIndex--;
            }
            else
            {
                _keypoints.at(monoIndex) = (*keypoint);
                desc.row(i).copyTo(descriptors.row(monoIndex));
                monoIndex++;
            }
            i++;
        }
        return monoIndex;
    }

} // namespace ORB_SLAM
