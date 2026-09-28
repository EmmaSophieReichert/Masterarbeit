import cv2
import numpy as np
import open3d as o3d
import pygame
import CONFIG

MESH_NUMBER_OF_POINTS = 200000

class ImageProjection:

    def __init__(self, calibration, logger=None):
        self.calibration = calibration
        self.logger = logger

        pygame.init()

        pygame.display.init()

        self.screen = None
        self.vertex_colors = None

    """
    Show the image on the projector in fullscreen mode
    """
    def show_img_fullscreen(self, img = None, procam_test_mode = False, on_image_shown=None, on_image_loop=None):

        num_displays = pygame.display.get_num_displays()
        target_display = 1 if num_displays > 1 else 0
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=target_display)
        
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

        print("PYGAME Screen size")
        print(self.screen.get_size())

        if(img is None):
            image = pygame.image.load("Projection/output.png")
            image = pygame.transform.scale(image, self.screen.get_size())
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

        if procam_test_mode:
            import select
            import sys
            print("Press enter for next image.")
        
        running = True
        shown = False
        while running:
            if on_image_loop:
                img = on_image_loop()
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

            self.screen.blit(image, (0, 0))
            pygame.display.flip()

            if not shown:
                pygame.event.pump()
                shown = True
                print("Image is shown on projector.")
                if on_image_shown:
                    on_image_shown()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            if procam_test_mode:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    input()  
                    running = False

        self.screen.fill((0, 0, 0))
        pygame.display.flip()

    """
    Bake the texture of the mesh into vertex colors
    """
    def bake_texture_to_vertices(self, mesh):

        print("Backe Textur in Vertex Colors...")

        pcd = mesh.sample_points_uniformly(number_of_points=MESH_NUMBER_OF_POINTS)

        if not pcd.has_colors():
            print("WARNUNG: Keine Farben in der Textur/Mesh gefunden! Nutze Standardfarbe Grau.")
            mesh.paint_uniform_color([0.5, 0.5, 0.5]) 
        else:
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            indices = [
                pcd_tree.search_knn_vector_3d(vert, 1)[1][0] 
                for vert in np.asarray(mesh.vertices)
            ]
            
            colors_3d = np.asarray(pcd.colors)
            vertex_colors = colors_3d[indices]

            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print("Fertig gebacken!")
        self.vertex_colors = np.asarray(
            mesh.vertex_colors
        ).copy()
        return mesh

    """
    Create a mesh projection of the given mesh and save it as an image
    """
    def create_mesh_projection(self, mesh, is_overlay=False, height=CONFIG.WINDOW_HEIGHT, width=CONFIG.WINDOW_WIDTH, save_img=True):
        print("Create Mesh Projection")
        if self.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        else:
            mesh = self.bake_texture_to_vertices(mesh)
        if mesh.has_vertex_colors():
            colors_bgr = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)[:, [2, 1, 0]]
        else:
            print("Keine Farben/Texturen gefunden! Nutze Grau.")
            colors_bgr = np.full((len(mesh.vertices), 3), (128, 128, 128), dtype=np.uint8)

        vertices_3d = np.asarray(mesh.vertices)
        if is_overlay:
            projected_verts = self.calibration.points_to_image(vertices_3d)
        else:
            projected_verts = self.calibration.points_to_projection(vertices_3d)
        faces = np.asarray(mesh.triangles)

        if is_overlay:
            proj_img = np.zeros((height, width, 3), dtype=np.uint8)
        else: 
            proj_img = np.zeros((CONFIG.WINDOW_HEIGHT, CONFIG.WINDOW_WIDTH, 3), dtype=np.uint8)
        z_depths = np.mean(vertices_3d[faces][:, :, 2], axis=1)
        sorted_indices = np.argsort(z_depths)[::-1]

        for idx in sorted_indices:
            face = faces[idx]
            pts = projected_verts[face].reshape((-1, 1, 2))
            
            color = colors_bgr[face[0]].tolist()
            cv2.fillPoly(proj_img, [pts.astype(np.int32)], color)

        # rectangle was needed sometimes, because projector would turn off with just little projection 
        #cv2.rectangle(proj_img, (0, 0), (int(proj_img.shape[1]*0.5), int(proj_img.shape[0]*0.5)), (255, 255, 255), thickness=50)
        if save_img:
            if is_overlay:
                cv2.imwrite("Projection/output_overlay.png", proj_img)
            else:
                cv2.imwrite("Projection/output.png", proj_img)
        print("Bild erfolgreich gespeichert!")
        return proj_img

    """
    Create a point cloud projection of the given point cloud and save it as an image
    """
    def create_pointcloud_projection(self, pcd, is_overlay=False,
                                 height=CONFIG.WINDOW_HEIGHT,
                                 width=CONFIG.WINDOW_WIDTH):

        points = np.asarray(pcd.points)

        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)[:, [2,1,0]]
        else:
            colors = np.full((len(points),3), 128, dtype=np.uint8)

        if is_overlay:
            projected = self.calibration.points_to_image(points)
            img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            projected = self.calibration.points_to_projection(points)
            img = np.zeros((CONFIG.WINDOW_HEIGHT,
                            CONFIG.WINDOW_WIDTH,3), dtype=np.uint8)

        order = np.argsort(points[:,2])[::-1]

        for i in order:
            x, y = projected[i]

            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (int(x), int(y)), 2, colors[i].tolist(), -1)
        cv2.imwrite("Projection/output.png", img)
        return img