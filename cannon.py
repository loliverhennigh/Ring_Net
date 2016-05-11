
import numpy as np
import random
import math

class Cannon:
    # A little ball trajectory simulation like the old Cannon games.
    # The data it generates can either be in the form of pixel images
    # of size 28x28 (like mnist!) or position of ball as two numbers 
    # between 0 and 1.

    def __init__(self):
        # ball starts at 0.5 and 0.0
        self.x_pos = .0 
        self.y_pos = .5 
        self.x_vel = random.random()
        self.y_vel = 5*random.random() 
        # you can play with this
        self.grav = 4.0
        self.dt = .01 
        self.damp = 0.01 #probably set this to 0 for most applications

    # very basic physics
    def restart(self):
        self.x_pos = .0 
        self.y_pos = .5 
        self.x_vel = random.random()
        self.y_vel = 5*(random.random()-.5) 

    # very basic physics
    def update_pos(self):
        for i in xrange(5):
            self.x_pos = self.x_pos + self.dt * self.x_vel
            self.y_pos = self.y_pos + self.dt * self.y_vel
            self.x_vel = self.x_vel + self.dt * (self.grav - (self.damp * self.x_vel))
            self.y_vel = self.y_vel + self.dt * self.damp * self.x_vel
            #self.y_vel = self.y_vel + self.dt * (self.grav - (self.damp * self.y_vel))
            # bounce
            if (0.1428 > self.x_pos):
                self.x_vel = -self.x_vel 
                self.x_pos = 0.143 
            if ((1-0.1428) < self.x_pos):
                self.x_vel = -self.x_vel 
                self.x_pos = (1-0.143)
            if (0.1428 > self.y_pos):
                self.y_vel = -self.y_vel 
                self.y_pos = 0.143 
            if ((1-0.1428) < self.y_pos):
                self.y_vel = -self.y_vel 
                self.y_pos = (1-0.143) 
 
    # generate pixell images
    def image_28x28(self):
        # same algorith as seen on
        # https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
        im = np.zeros((28,28))
        radius = 4
        x0 = (self.x_pos * 28) // 1
        y0 = (self.y_pos * 28) // 1
        x = radius  
        y = 0
        decisionOver2 = 1 - x
        while(y <= x):
            im[(x+x0) % 28, (y+y0) % 28] = 1.0
            im[(y+x0) % 28, (x+y0) % 28] = 1.0
            im[(-x+x0) % 28, (y+y0) % 28] = 1.0
            im[(-y+x0) % 28, (x+y0) % 28] = 1.0
            im[(-x+x0) % 28, (-y+y0) % 28] = 1.0
            im[(-y+x0) % 28, (-x+y0) % 28] = 1.0
            im[(x+x0) % 28, (-y+y0) % 28] = 1.0
            im[(y+x0) % 28, (-x+y0) % 28] = 1.0
            y = y + 1
            if (decisionOver2 <=0):
                decisionOver2 = decisionOver2 + 2*y + 1
            else:
                x = x - 1
                decisionOver2 = decisionOver2 + 2*(y-x) + 1

        # now draw vel 
        """radius = 2
        x0 = (self.x_vel * 4) // 1 
        y0 = ((self.y_vel * 4) // 1) + 14
        x = radius  
        y = 0
        decisionOver2 = 1 - x
        while(y <= x):
            im[(x+x0) % 28, (y+y0) % 28] = 1.0
            im[(y+x0) % 28, (x+y0) % 28] = 1.0
            im[(-x+x0) % 28, (y+y0) % 28] = 1.0
            im[(-y+x0) % 28, (x+y0) % 28] = 1.0
            im[(-x+x0) % 28, (-y+y0) % 28] = 1.0
            im[(-y+x0) % 28, (-x+y0) % 28] = 1.0
            im[(x+x0) % 28, (-y+y0) % 28] = 1.0
            im[(y+x0) % 28, (-x+y0) % 28] = 1.0
            y = y + 1
            if (decisionOver2 <=0):
                decisionOver2 = decisionOver2 + 2*y + 1
            else:
                x = x - 1
                decisionOver2 = decisionOver2 + 2*(y-x) + 1"""
        return im 

    def generate_28x28(self, batch_size, num_steps):
        # makes a np array of size batch_size x num_steps
        # this will be what most learing algorithms need
        # from there data
        x_0 = np.zeros([batch_size, num_steps, 28, 28])
        x_1 = np.zeros([batch_size, num_steps, 28, 28])
        x_2 = np.zeros([batch_size, num_steps, 28, 28])
        for i in xrange(batch_size):
            self.restart()
            x_0[i, 0, :] = self.image_28x28()
            self.update_pos()
            x_0[i, 1, :] = self.image_28x28()
            x_1[i, 0, :] = self.image_28x28()
            for j in xrange(num_steps-2):
                #time.sleep(.5)
                self.update_pos()
                x_0[i, j + 2, :] = self.image_28x28()
                x_1[i, j + 1, :] = x_0[i, j + 2, :] 
                x_2[i, j, :] = x_0[i, j + 2, :] 
            self.update_pos()
            x_1[i, num_steps-1, :] = self.image_28x28()
            x_2[i, num_steps-2, :] = self.image_28x28()
            self.update_pos()
            x_2[i, num_steps-1, :] = self.image_28x28()
        return x_0, x_1, x_2

    def speed(self):
        return math.sqrt(self.x_vel ** 2 + self.y_vel ** 2)

if __name__ == "__main__":
    k = Cannon()




