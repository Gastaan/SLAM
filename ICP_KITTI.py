import copy
import open3d as o3d
import numpy as np
from natsort import natsorted
import os
import time


def convert_kitti_bin_to_pcd(binFilePath):
    # Load binary point cloud
    bin_pcd = np.fromfile(binFilePath, dtype=np.float32)

    # Reshape and drop reflection values, We drop the intensity column
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Convert to Open3D point cloud
    return pcd


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def downsample_point_cloud(pcd, voxel_size=0.2):
    return pcd.voxel_down_sample(voxel_size)


def save_point_cloud(pcd, filename):
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def iterative_closest_point(source, target):
    threshold = 5  # Increase the threshold
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4],
                             [0.0, 0.0, 0.0, 1.0]])

    # Downsample the point clouds
    source_down = downsample_point_cloud(source)
    target_down = downsample_point_cloud(target)

    print("Apply point-to-point ICP")

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=1500
        )
    )

    if not reg_p2p.transformation.any():
        print("ICP did not converge.")
        return None

    print("ICP Result:")
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    # draw_registration_result(source, target, reg_p2p.transformation)

    return reg_p2p.transformation


input_dir = "kitti_sample/00/"
input_files = natsorted(os.listdir(input_dir))

# read first bin file
pcd = convert_kitti_bin_to_pcd(os.path.join(input_dir, input_files[0]))

# getting open3d to display the video
vis = o3d.visualization.Visualizer()
vis.create_window()

# iterate through remaining files
for i in range(1, len(input_files)):
    input_file_path = os.path.join(input_dir, input_files[i])

    # remove previous pointcloud and add new one
    vis.remove_geometry(pcd, False)

    new_pcd = convert_kitti_bin_to_pcd(input_file_path)

    transformation = iterative_closest_point(source=pcd, target=new_pcd)
    if transformation is not None:
        pcd.transform(transformation)
        pcd = pcd + new_pcd

    # add pcd to visualizer
    vis.add_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.005)  # makes display a little smoother (display will not match fps of video)
    save_point_cloud(pcd, f"map_snapshots/output_pcd_{i}.pcd")

vis.destroy_window()
save_point_cloud(pcd, "final_output.pcd")
