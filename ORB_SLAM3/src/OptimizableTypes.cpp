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

#include "OptimizableTypes.h"

namespace ORB_SLAM3
{
    bool EdgeSE3ProjectXYZOnlyPose::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZOnlyPose::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        if (pCamera->GetType() == 2)
        {
            int cols_ = pCamera->getParameter(0);
            int rows_ = pCamera->getParameter(1);
            std::cout << "___________jac1____________\n";
            const auto pcx = xyz_trans[0];
            const auto pcy = xyz_trans[1];
            const auto pcz = xyz_trans[2];
            const auto L = xyz_trans.norm();

            // 回転に対する微分
            const Eigen::Vector3d d_pc_d_rx(0, -pcz, pcy);
            const Eigen::Vector3d d_pc_d_ry(pcz, 0, -pcx);
            const Eigen::Vector3d d_pc_d_rz(-pcy, pcx, 0);
            // 並進に対する微分
            const Eigen::Vector3d d_pc_d_tx(1, 0, 0);
            const Eigen::Vector3d d_pc_d_ty(0, 1, 0);
            const Eigen::Vector3d d_pc_d_tz(0, 0, 1);

            // 状態ベクトルを x = [rx, ry, rz, tx, ty, tz] として，
            // 導関数ベクトル d_pcx_d_x, d_pcy_d_x, d_pcz_d_x を作成
            Eigen::Matrix<double, 6, 1> d_pcx_d_x;
            d_pcx_d_x << d_pc_d_rx(0), d_pc_d_ry(0), d_pc_d_rz(0),
                d_pc_d_tx(0), d_pc_d_ty(0), d_pc_d_tz(0);
            Eigen::Matrix<double, 6, 1> d_pcy_d_x;
            d_pcy_d_x << d_pc_d_rx(1), d_pc_d_ry(1), d_pc_d_rz(1),
                d_pc_d_tx(1), d_pc_d_ty(1), d_pc_d_tz(1);
            Eigen::Matrix<double, 6, 1> d_pcz_d_x;
            d_pcz_d_x << d_pc_d_rx(2), d_pc_d_ry(2), d_pc_d_rz(2),
                d_pc_d_tx(2), d_pc_d_ty(2), d_pc_d_tz(2);

            // 導関数ベクトル d_L_d_x を作成
            const Eigen::Matrix<double, 6, 1> d_L_d_x = (1.0 / L) * (pcx * d_pcx_d_x + pcy * d_pcy_d_x + pcz * d_pcz_d_x);

            // ヤコビ行列を作成
            Eigen::Matrix<double, 2, 6> jacobian = Eigen::Matrix<double, 2, 6>::Zero();
            jacobian.block<1, 6>(0, 0) = -(cols_ / (2 * M_PI)) * (1.0 / (pcx * pcx + pcz * pcz)) * (pcz * d_pcx_d_x - pcx * d_pcz_d_x);
            jacobian.block<1, 6>(1, 0) = -(rows_ / M_PI) * (1.0 / (L * std::sqrt(pcx * pcx + pcz * pcz))) * (L * d_pcy_d_x - pcy * d_L_d_x);

            // g2oの変数にセット
            // 姿勢に対する微分
            _jacobianOplusXi = jacobian;
        }
        else{

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
            -z, 0.f, x, 0.f, 1.f, 0.f,
            y, -x, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(xyz_trans) * SE3deriv;}
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZOnlyPoseToBody::linearizeOplus()
    {
        std::cout << "___________jac2____________\n";
        g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T_lw(vi->estimate());
        Eigen::Vector3d X_l = T_lw.map(Xw);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(Xw));

        double x_w = X_l[0];
        double y_w = X_l[1];
        double z_w = X_l[2];

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z_w, -y_w, 1.f, 0.f, 0.f,
            -z_w, 0.f, x_w, 0.f, 1.f, 0.f,
            y_w, -x_w, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }

    EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
    {
    }

    bool EdgeSE3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZ::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZ::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T(vj->estimate());
        g2o::VertexSBAPointXYZ *vi = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
        Eigen::Vector3d xyz = vi->estimate();
        Eigen::Vector3d xyz_trans = T.map(xyz);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        
        if (pCamera->GetType() == 2)
        {
            Eigen::Matrix3d rot_cw;
            rot_cw << T.rotation().toRotationMatrix();

            int cols_ = pCamera->getParameter(0);
            int rows_ = pCamera->getParameter(1);
            const auto pcx = x;
            const auto pcy = y;
            const auto pcz = z;
            const auto L = xyz_trans.norm();

            // 回転に対する微分
            const Eigen::Vector3d d_pc_d_rx(0, -pcz, pcy);
            const Eigen::Vector3d d_pc_d_ry(pcz, 0, -pcx);
            const Eigen::Vector3d d_pc_d_rz(-pcy, pcx, 0);
            // 並進に対する微分
            const Eigen::Vector3d d_pc_d_tx(1, 0, 0);
            const Eigen::Vector3d d_pc_d_ty(0, 1, 0);
            const Eigen::Vector3d d_pc_d_tz(0, 0, 1);
            // 3次元点に対する微分
            const Eigen::Vector3d d_pc_d_pwx = rot_cw.block<3, 1>(0, 0);
            const Eigen::Vector3d d_pc_d_pwy = rot_cw.block<3, 1>(0, 1);
            const Eigen::Vector3d d_pc_d_pwz = rot_cw.block<3, 1>(0, 2);

            // 状態ベクトルを x = [rx, ry, rz, tx, ty, tz, pwx, pwy, pwz] として，
            // 導関数ベクトル d_pcx_d_x, d_pcy_d_x, d_pcz_d_x を作成
            Eigen::Matrix<double, 9, 1> d_pcx_d_x;
            d_pcx_d_x << d_pc_d_rx(0), d_pc_d_ry(0), d_pc_d_rz(0),
                d_pc_d_tx(0), d_pc_d_ty(0), d_pc_d_tz(0),
                d_pc_d_pwx(0), d_pc_d_pwy(0), d_pc_d_pwz(0);
            Eigen::Matrix<double, 9, 1> d_pcy_d_x;
            d_pcy_d_x << d_pc_d_rx(1), d_pc_d_ry(1), d_pc_d_rz(1),
                d_pc_d_tx(1), d_pc_d_ty(1), d_pc_d_tz(1),
                d_pc_d_pwx(1), d_pc_d_pwy(1), d_pc_d_pwz(1);
            Eigen::Matrix<double, 9, 1> d_pcz_d_x;
            d_pcz_d_x << d_pc_d_rx(2), d_pc_d_ry(2), d_pc_d_rz(2),
                d_pc_d_tx(2), d_pc_d_ty(2), d_pc_d_tz(2),
                d_pc_d_pwx(2), d_pc_d_pwy(2), d_pc_d_pwz(2);

            // 導関数ベクトル d_L_d_x を作成
            const Eigen::Matrix<double, 9, 1> d_L_d_x = (1.0 / L) * (pcx * d_pcx_d_x + pcy * d_pcy_d_x + pcz * d_pcz_d_x);

            // ヤコビ行列を作成
            Eigen::Matrix<double, 2, 9> jacobian = Eigen::Matrix<double, 2, 9>::Zero();
            jacobian.block<1, 9>(0, 0) = -(cols_ / (2 * M_PI)) * (1.0 / (pcx * pcx + pcz * pcz)) * (pcz * d_pcx_d_x - pcx * d_pcz_d_x);
            jacobian.block<1, 9>(1, 0) = -(rows_ / M_PI) * (1.0 / (L * std::sqrt(pcx * pcx + pcz * pcz))) * (L * d_pcy_d_x - pcy * d_L_d_x);

            // g2oの変数にセット
            // 3次元点に対する微分
            _jacobianOplusXi = jacobian.block<2, 3>(0, 6);
            // 姿勢に対する微分
            _jacobianOplusXj = jacobian.block<2, 6>(0, 0);
            
        }
        else
        {
            auto projectJac = -pCamera->projectJac(xyz_trans);

            _jacobianOplusXi = projectJac * T.rotation().toRotationMatrix();

            Eigen::Matrix<double, 3, 6> SE3deriv;
            SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
                -z, 0.f, x, 0.f, 1.f, 0.f,
                y, -x, 0.f, 0.f, 0.f, 1.f;

            _jacobianOplusXj = projectJac * SE3deriv;
        }
    }

    EdgeSE3ProjectXYZToBody::EdgeSE3ProjectXYZToBody() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
    {
    }

    bool EdgeSE3ProjectXYZToBody::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZToBody::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZToBody::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T_lw(vj->estimate());
        g2o::SE3Quat T_rw = mTrl * T_lw;
        g2o::VertexSBAPointXYZ *vi = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
        Eigen::Vector3d X_w = vi->estimate();
        Eigen::Vector3d X_l = T_lw.map(X_w);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(X_w));

        _jacobianOplusXi = -pCamera->projectJac(X_r) * T_rw.rotation().toRotationMatrix();

        double x = X_l[0];
        double y = X_l[1];
        double z = X_l[2];

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
            -z, 0.f, x, 0.f, 1.f, 0.f,
            y, -x, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }

    VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, g2o::Sim3>()
    {
        _marginalized = false;
        _fix_scale = false;
    }

    bool VertexSim3Expmap::read(std::istream &is)
    {
        g2o::Vector7d cam2world;
        for (int i = 0; i < 6; i++)
        {
            is >> cam2world[i];
        }
        is >> cam2world[6];

        float nextParam;
        for (size_t i = 0; i < pCamera1->size(); i++)
        {
            is >> nextParam;
            pCamera1->setParameter(nextParam, i);
        }

        for (size_t i = 0; i < pCamera2->size(); i++)
        {
            is >> nextParam;
            pCamera2->setParameter(nextParam, i);
        }

        setEstimate(g2o::Sim3(cam2world).inverse());
        return true;
    }

    bool VertexSim3Expmap::write(std::ostream &os) const
    {
        g2o::Sim3 cam2world(estimate().inverse());
        g2o::Vector7d lv = cam2world.log();
        for (int i = 0; i < 7; i++)
        {
            os << lv[i] << " ";
        }

        for (size_t i = 0; i < pCamera1->size(); i++)
        {
            os << pCamera1->getParameter(i) << " ";
        }

        for (size_t i = 0; i < pCamera2->size(); i++)
        {
            os << pCamera2->getParameter(i) << " ";
        }

        return os.good();
    }

    EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeSim3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSim3ProjectXYZ::write(std::ostream &os) const
    {
        for (int i = 0; i < 2; i++)
        {
            os << _measurement[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ() : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeInverseSim3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeInverseSim3ProjectXYZ::write(std::ostream &os) const
    {
        for (int i = 0; i < 2; i++)
        {
            os << _measurement[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

}
