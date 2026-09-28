import cv2
import open3d as o3d
import numpy as np
from Tracking.PPF_faster import *
import CONFIG

#For TEASER++:
#import _teaserpp
#import scipy.spatial as spatial

class AutomaticRegistration:

    def __init__(self, noise_bound=7.0):
        # Only for TEASER**:
        self.noise_bound = noise_bound
        self.voxel_size = 5.0
        self.radius_multiplier = 5
        self.filtering_percentile = 25

    """
        Apply PPF from whateverforever and ICP
    """
    def get_registration_matrix_with_ppf(self, source_pcd_dense, target_pcd_dense, with_icp=True, voxel_size=CONFIG.VOXEL_SIZE):

        source_pcd = source_pcd_dense.voxel_down_sample(voxel_size=voxel_size)
        target_pcd = target_pcd_dense.voxel_down_sample(voxel_size=voxel_size)

        print(len(source_pcd_dense.points))
        print(len(source_pcd.points))

        #o3d.visualization.draw_geometries([source_pcd])
        #o3d.visualization.draw_geometries([target_pcd])

        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=CONFIG.VOXEL_SIZE * 2, max_nn=30
        ))
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=CONFIG.VOXEL_SIZE * 2, max_nn=30
        ))
        target_pcd.orient_normals_consistent_tangent_plane(k=30) #inverts normals, because they are the wrong way around

        # o3d.visualization.draw_geometries([source_pcd], window_name="Source PCD", 
        #                               point_show_normal=True)
        # o3d.visualization.draw_geometries([target_pcd], window_name="Target PCD", 
        #                               point_show_normal=True)

        o3d.io.write_point_cloud("temp_model.ply", source_pcd)
        o3d.io.write_point_cloud("temp_scan.ply", target_pcd)

        all_poses = get_poses("temp_model.ply", "temp_scan.ply", scene_pts_fraction=1.0, ppf_rel_dist_step=0.05)#0.05)#0.08) #Emma
        print(f"PPF beendet. Anzahl gefundener Posen: {len(all_poses)}")
        all_icp = []
        if all_poses:
            if with_icp:
                for pose in all_poses:
                    matrix = pose[0]
                    print("\n--- Matrix vor ICP ---")
                    print(np.array2string(matrix, precision=4, suppress_small=True))

                    for threshold in [CONFIG.VOXEL_SIZE]:
                        print(f"Running ICP with threshold: {threshold}")
                        final_matrix, fitness, rmse = self.refine_registration_icp(
                            source_pcd_dense, target_pcd_dense, matrix, threshold=threshold
                        )
                        # final_matrix, fitness, rmse =self.apply_multi_scale_icp(source_pcd_dense, target_pcd_dense, matrix) #if you want multi-scale instead
                        print(f"ICP beendet. Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")
                        matrix = final_matrix

                    # optional: evaluate by full point clouds (not downsampled)
                    # eval_result = o3d.pipelines.registration.evaluate_registration(
                    #     source_pcd_dense, target_pcd_dense, max_correspondence_distance=2.5, transformation=matrix
                    # )
                    
                    # fitness_dense = eval_result.fitness
                    # rmse_dense = eval_result.inlier_rmse

                    # print(f"ICP beendet. Fitness (dicht): {fitness_dense:.4f}, RMSE (dicht): {rmse_dense:.4f}")

                    all_icp.append([matrix, pose[1], pose[2], fitness, rmse])

                    print("\n--- Finale Matrix nach ICP ---")
                    print(np.array2string(matrix, precision=4, suppress_small=True))
                all_icp.sort(key=lambda x: x[4]) #take best icp value first, not from ppf
                return all_icp
            return all_poses
        print("PPF: Keine Pose gefunden. Identitätsmatrix wird verwendet.")
        return np.eye(4)
    
    """
    Applies ICP to found position
    """
    def refine_registration_icp(self, source, target, initial_trans, threshold=CONFIG.VOXEL_SIZE, iterations=50):

        if not source.has_normals(): source.estimate_normals() 
        if not target.has_normals(): target.estimate_normals()

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, initial_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        )
        
        print(f"ICP beendet. Fitness: {reg_p2p.fitness:.4f}, RMSE: {reg_p2p.inlier_rmse:.4f}")
        return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

    #---------------------------------------------------------------------------------------------------
    #Alternative algorithms for registration (not used in final version)
    #---------------------------------------------------------------------------------------------------

    """
        Calculates FPFH-features for TEASER++ and FPFH registration
    """
    def preprocess_point_cloud(self, pcd): 
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * self.radius_multiplier, max_nn=100)
        )
        return pcd_down, fpfh
    

    def print_pcd_info(self, name, pcd):
        points = np.asarray(pcd.points)
        num_points = len(points)
        if num_points == 0:
            print(f" WARNUNG: {name} ist leer!")
            return
        
        min_b = pcd.get_min_bound()
        max_b = pcd.get_max_bound()
        dim = max_b - min_b
        
        print(f"--- Info für {name} ---")
        print(f"Anzahl Punkte: {num_points}")
        print(f"Dimensionen (X,Y,Z): {dim[0]:.4f}, {dim[1]:.4f}, {dim[2]:.4f}")
        print(f"Schwerpunkt: {pcd.get_center()}")
        print("-" * 25)

    def visualize_inlier_lines(self, source_points, target_points, inlier_indices):
        """
        source_points, target_points: Arrays der Form [3, N]
        inlier_indices: Liste der vom Solver gefundenen Inlier-Indizes
        """
        src_inliers = source_points[:, inlier_indices].T
        tgt_inliers = target_points[:, inlier_indices].T
        
        all_points = np.concatenate([src_inliers, tgt_inliers], axis=0)
        
        lines = [[i, i + len(inlier_indices)] for i in range(len(inlier_indices))]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))]) # Grüne Linien
        
        return line_set

    """
    Use TEASER++ to find registration matrix
    make sure to build teaser and import it
    Link: https://github.com/MIT-SPARK/TEASER-plusplus
    """
    def get_registration_matrix_with_teaser(self, source_pcd, target_pcd, scaling=False):

        self.print_pcd_info("Source (Hase)", source_pcd)
        self.print_pcd_info("Target (Tiefenbild)", target_pcd)

        target_pcd.estimate_normals()
        target_pcd.orient_normals_consistent_tangent_plane(k=20)

        source_pcd.estimate_normals()

        src_down, src_fpfh = self.preprocess_point_cloud(source_pcd)
        tgt_down, tgt_fpfh = self.preprocess_point_cloud(target_pcd)

        self.print_pcd_info("Source (Hase) nach Preprocess", src_down)
        self.print_pcd_info("Target (Tiefenbild) nach Preprocess", tgt_down)

        # #Show how many points are left
        # o3d.visualization.draw_geometries([tgt_down, src_down])
        # return 

        tree = spatial.KDTree(tgt_fpfh.data.T)
        distances, idx = tree.query(src_fpfh.data.T)

        threshold = np.percentile(distances, self.filtering_percentile) # filter the best matches
        mask = distances < threshold

        # Print match quality statistics
        num_total_possible = len(distances)
        num_selected = np.sum(mask)

        print(f"FPFH Matching Statistik:")
        print(f" - Mögliche Paare insgesamt: {num_total_possible}")
        print(f" - Ausgewählte Paare (Top 25%): {num_selected}")
        print(f" - Durchschnittliche Feature-Distanz: {np.mean(distances):.4f}")
        print(f" - Durchschnittliche Feature-Distanz (selected): {np.mean(distances[mask]):.4f}")
        print(f" - Minimale Distanz: {np.min(distances):.4f}")
        print(f" - Maximale Distanz: {np.max(distances):.4f}")

        src_corr = np.asarray(src_down.points).T[:, mask]
        tgt_corr = np.asarray(tgt_down.points)[idx].T[:, mask]
        # src_corr = np.asarray(src_down.points).T
        # tgt_corr = np.asarray(tgt_down.points).T

        solver_params = _teaserpp.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1.0 # Standard aus Quickstart
        
        solver_params.noise_bound = self.noise_bound
        
        solver_params.estimate_scaling = scaling
        solver_params.rotation_estimation_algorithm = (
            _teaserpp.RotationEstimationAlgorithm.GNC_TLS
        )
        
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 1000
        
        solver = _teaserpp.RobustRegistrationSolver(solver_params)
        solver.solve(src_corr, tgt_corr)
        solution = solver.getSolution()

        inlier_indices = solver.getInlierMaxClique()
        num_inliers = len(inlier_indices)
        print(f"Anzahl der Inlier: {len(inlier_indices)}")

        # build matrix 
        transformation = np.eye(4)
        if scaling:
            transformation[:3, :3] = solution.scale * solution.rotation
        else:
            transformation[:3, :3] = solution.rotation
        transformation[:3, 3] = solution.translation

        # # Visualize line graph
        # lines = self.visualize_inlier_lines(src_corr, tgt_corr, inlier_indices)
        # o3d.visualization.draw_geometries([target_pcd, source_pcd, lines])
        
        model_idx = 0
        score = num_inliers
        all_registrations = [[transformation, model_idx, score]]

        return all_registrations

    """
    Use FPFH features to find registration matrix
    """
    def get_registration_matrix_with_fpfh(self, source_pcd, target_pcd):
        voxel_size = 14.0
        
        def preprocess_point_cloud_2(pcd, voxel_size):
            pcd_down = pcd.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
            return pcd_down, fpfh

        source_down, source_fpfh = preprocess_point_cloud_2(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud_2(target_pcd, voxel_size)

        # RANSAC matching (Global Registration)
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        return [[result.transformation, 0, 0]]

    """
    Use PPF from OpenCV to find registration matrix
    """
    def get_registration_matrix_with_ppf_2(self, source_pcd, target_pcd, with_icp=True, voxel_size = CONFIG.VOXEL_SIZE):
        source_pcd = source_pcd.voxel_down_sample(voxel_size=voxel_size)
        target_pcd = target_pcd.voxel_down_sample(voxel_size=voxel_size)

        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=CONFIG.VOXEL_SIZE * 2, max_nn=30
        ))
        
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=CONFIG.VOXEL_SIZE * 2, max_nn=30
        ))
        target_pcd.orient_normals_consistent_tangent_plane(k=30) #inverts normals, because they are the wrong way around - check needed?
        
        # o3d.visualization.draw_geometries([source_pcd], window_name="Source PCD", 
        #                               point_show_normal=True)
        # o3d.visualization.draw_geometries([target_pcd], window_name="Target PCD", 
        #                               point_show_normal=True)

        def o3d_to_cv_pc(pcd):
            pts = np.asarray(pcd.points, dtype=np.float32)
            normals = np.asarray(pcd.normals, dtype=np.float32)
            return np.hstack((pts, normals))

        try:
            cv_model = o3d_to_cv_pc(source_pcd)
            cv_scene = o3d_to_cv_pc(target_pcd)
        except ValueError:
            print("Fehler: Punktwolken konnten nicht konvertiert werden.")
            return np.eye(4)

        # Parameter: relative_sampling_step (Standard: 0.025), relative_distance_step (Standard: 0.05)
        detector = cv2.ppf_match_3d_PPF3DDetector(0.025, 0.01)
        
        print("Trainiere OpenCV PPF-Modell...")
        detector.trainModel(cv_model)

        print("Starte OpenCV PPF-Matching auf der Szene...")
        # Parameter: scene, relativeSceneSampleStep (1/5 der Szene prüfen), sceneRadiusFraction
        results = detector.match(cv_scene, 1.0/1.0, 0.025)
        
        print(f"PPF beendet. Anzahl roher Posen: {len(results)}")

        if not results:
            print("PPF: Keine Pose gefunden. Identitätsmatrix wird verwendet.")
            return np.eye(4)

        if with_icp:
            icp = cv2.ppf_match_3d_ICP(100)
            
            num_top_poses = min(3, len(results))
            retval, refined_results = icp.registerModelToScene(cv_model, cv_scene, results[:num_top_poses])
            
            output_poses = refined_results if retval == 0 else results[:num_top_poses]
            for i, p in enumerate(output_poses):
                print(f"--- Pose {i+1} ---")
                print(f"RMSE (Residual):  {p.residual:.5f}")
                print(f"Votes (Fitness):  {p.numVotes}")
                print(f"Transformationsmatrix:\n{p.pose}\n")
        else:
            output_poses = results[:3]

        all_icp = []
        for i, result in enumerate(output_poses):
            matrix = np.array(result.pose) 
            score = getattr(result, 'numVotes', 0)
            model_idx = getattr(result, 'modelIndex', 0)

            print(f"\n--- Finale Matrix #{i+1} (Score/Votes: {score}) ---")
            print(np.array2string(matrix, precision=4, suppress_small=True))
            
            all_icp.append([matrix, model_idx, score])

        return all_icp

    """
    Applies GICP to found position
    Install small_gicp via 'pip install small-gicp' first
    """
    def refine_registration_gicp(self, source_o3d, target_o3d, initial_trans, threshold=CONFIG.VOXEL_SIZE, iterations=150):

        import small_gicp, copy
        target, target_tree = small_gicp.preprocess_points(
            np.asarray(target_o3d.points, dtype=np.float64),
            downsampling_resolution=-1.0,
        )
        source, _ = small_gicp.preprocess_points(
            np.asarray(source_o3d.points, dtype=np.float64),
            downsampling_resolution=-1.0,
        )

        result = small_gicp.align(
            target=target,
            source=source,
            target_tree=target_tree,
            init_T_target_source=np.asarray(initial_trans, dtype=np.float64),
            registration_type="GICP",  # "GICP", "VGICP", "PLANE_TO_PLANE", "POINT_TO_POINT"
            max_correspondence_distance=threshold,
            max_iterations=iterations,
            num_threads=4,
        )

        final_matrix = result.T_target_source

        # OPTIONAL: Visualize

        # source_vis = copy.deepcopy(source_o3d)
        # source_vis.transform(final_matrix)

        # target_o3d.paint_uniform_color([0.6, 0.6, 0.6]) 
        # source_vis.paint_uniform_color([0.0, 0.8, 0.8]) 
        # print("Visualisierung: Target=Grau, Source=Cyan")

        # o3d.visualization.draw_geometries([target_o3d, source_vis])

        converged = result.converged
        fitness = 1.0 if converged else 0.0
        rmse = result.error

        return final_matrix, fitness, rmse

    """
    Applies multi-scale ICP to found position
    from: https://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html#Multi-Scale-ICP-Example
    """
    def apply_multi_scale_icp(self, source, target, init_matrix=None):
        treg = o3d.t.pipelines.registration

        if isinstance(source, o3d.geometry.PointCloud):
            source = o3d.t.geometry.PointCloud.from_legacy(source)
        if isinstance(target, o3d.geometry.PointCloud):
            target = o3d.t.geometry.PointCloud.from_legacy(target)

        init_tensor = (
            o3d.core.Tensor(init_matrix, dtype=o3d.core.Dtype.Float64)
            if init_matrix is not None
            else o3d.core.Tensor.eye(4, o3d.core.Dtype.Float64)
        )

        # Multi-scale configuration (Coarse-to-Fine)
        voxel_sizes = o3d.utility.DoubleVector([16.0, 10.0, 7.5, 5.0, 2.0])
        distances = o3d.utility.DoubleVector([45.0, 30.0, 22.5, 15.0, 6.0])
        criteria = [
            treg.ICPConvergenceCriteria(1e-3, 1e-3, max_iteration=50),
            treg.ICPConvergenceCriteria(1e-4, 1e-4, max_iteration=40),
            treg.ICPConvergenceCriteria(1e-5, 1e-5, max_iteration=30),
            treg.ICPConvergenceCriteria(1e-6, 1e-6, max_iteration=20),
            treg.ICPConvergenceCriteria(1e-7, 1e-7, max_iteration=10),
        ]

        estimation = treg.TransformationEstimationPointToPlane()

        callback_after_iteration = lambda loss_log_map : print("Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
            loss_log_map["iteration_index"].item(),
            loss_log_map["scale_index"].item(),
            loss_log_map["scale_iteration_index"].item(),
            loss_log_map["fitness"].item(),
            loss_log_map["inlier_rmse"].item()))

        result = treg.multi_scale_icp(
            source, target, voxel_sizes, criteria, distances, init_tensor, estimation, callback_after_iteration
        )

        def draw_registration_result(source, target, transformation):
            source_temp = source.clone()
            target_temp = target.clone()

            source_temp.transform(transformation)
            o3d.visualization.draw_geometries(
                [source_temp.to_legacy(),
                target_temp.to_legacy()],
                zoom=0.4459,
                front=[0.9288, -0.2951, -0.2242],
                lookat=[1.6784, 2.0612, 1.4451],
                up=[-0.3402, -0.9189, -0.1996])

        #draw_registration_result(source, target, result.transformation)

        return (
            result.transformation.numpy(),
            float(result.fitness),
            float(result.inlier_rmse),
        )