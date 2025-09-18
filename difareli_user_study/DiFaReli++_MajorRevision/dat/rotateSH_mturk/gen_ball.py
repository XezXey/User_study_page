import numpy as np
import torch as pt
from torchvision.utils import save_image

def applySHlight(normal_images, sh_coeff):
  N = normal_images
  sh = pt.stack(
    [
      N[0] * 0.0 + 1.0,
      N[0],
      N[1],
      N[2],
      N[0] * N[1],
      N[0] * N[2],
      N[1] * N[2],
      N[0] ** 2 - N[1] ** 2,
      3 * (N[2] ** 2) - 1,
    ],
    0,
  )  # [9, h, w]
  pi = np.pi
  constant_factor = pt.tensor(
    [
      1 / np.sqrt(4 * pi),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
      (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
    ]
  ).float()
  sh = sh * constant_factor[:, None, None]

  shading = pt.sum(
    sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
  )  # [9, 3, h, w]

  return shading

def applySHlightXYZ(xyz, sh):
  out = applySHlight(xyz, sh)
  # out /= pt.max(out)
  out *= 0.7
  return pt.clip(out, 0, 1)


def genSurfaceNormals(n):
  x = pt.linspace(-1, 1, n)
  y = pt.linspace(1, -1, n)
  y, x = pt.meshgrid(y, x)

  z = (1 - x ** 2 - y ** 2)
  z[z < 0] = 0
  z = pt.sqrt(z)
  return pt.stack([x, y, z], 0)

def drawSH(sh_np, output):
    sh = pt.tensor(sh_np).view(9, 3) 
    ball = drawSphere(sh)
    save_image(ball, output)
  
  
def drawSphere(sh):
  xyz = genSurfaceNormals(256)
  out = applySHlightXYZ(xyz, sh)
  out[:, xyz[2] == 0] = 0
  return out

if __name__ == '__main__':
    sh_text = "3.7467763 3.7607439 3.7748303 -0.17061733 -0.18169762 -0.18631022 0.07023415 0.07821477 0.07901665 -0.41905978 -0.40052396 -0.3687197 0.1470597 0.14538132 0.14575471 -0.28285792 -0.294153 -0.29541978 0.6458615 0.65194905 0.65252924 0.63551205 0.64745045 0.6551203 0.049085654 0.04447888 0.051973432"
    sh_np = np.array([float(x) for x in sh_text.split(" ")])
    drawSH(sh_np, f"original.png")
    
    