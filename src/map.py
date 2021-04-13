import pangolin
import OpenGL.GL as gl
import numpy as np

from multiprocessing import Process, Queue


class Map:
    def __init__(self, W, H):
        self.width = W
        self.Height = H
        self.poses = []
        self.locs = []
        self.gts = []
        self.points = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def add_observation(self, pose, gt, points):
        self.locs.append(pose[:3, 3])
        self.poses.append(pose)
        self.gts.append(gt[:3, 3])
        for point in points:
            self.points.append(point)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', int(self.width), int(self.Height))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(int(self.width), int(self.Height), 420, 420, int(self.width//2), int(self.Height//2), 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0,   0,  0,
                                     0,  -1,  0))
        self.handler = pangolin.Handler3D(self.scam)
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.width/self.Height)
        self.dcam.SetHandler(self.handler)

    def viewer_thread(self, q):
        self.viewer_init()
        while True:
            self.viewer_refresh(q)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        # draw trajectories
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawLines(self.state[1][:-1], self.state[1][1:], point_size=5)

        # draw gt
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawLines(self.state[2][:-1], self.state[2][1:], point_size=5)

        # draw keypoints
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[3])

        pangolin.FinishFrame()

    def display(self):
        poses = np.array(self.poses)
        points = np.array(self.points)
        locs = np.array(self.locs)
        gts = np.array(self.gts)
        self.q.put((poses, locs, gts, points))