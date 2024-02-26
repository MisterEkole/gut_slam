'''
CamTClin
'''
import numpy as  np
from utils import create_cylindrical_mesh, adjust_cylindrical_mesh, display_cylindrical_mesh_proj
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon

img=np.array(Image.open('/Users/ekole/Dev/gut_slam/gut_images/image1.jpeg'))

#init params
nb_theta=36
nb_h=10
R=20 #cylinder radius
f=1 
L=50
zmin=10 

#compute cylindrical mesh
theta,h, nappe,_=create_cylindrical_mesh(nb_pts_radiaux=nb_theta,nb_pts_axiaux=nb_h,rayon=R,hauteur=L)

print(theta.shape, h.shape, nappe.shape)

#camera params
vp=np.array([img.shape[0]/2,img.shape[1]/2,1]) #vanishing point
p_0=np.array([img.shape[0]/2,img.shape[1]/2,0]) #cylinder axis origin
umax, vmax=img.shape[0],img.shape[1] #image dimensions

K=np.array([[f,0,umax/2],[0,f,vmax/2],[0,0,1]]) #camera intrinsic matrix


#init mesh adjustment loop

fig, ax=plt.subplots()
plt.imshow(img)
plt.title('Cylindrical Mesh Projection')

plt.plot(vp[0],vp[1],'ro', markersize=10, linewidth=2)
plt.plot(p_0[0],p_0[1],'bo', markersize=10, linewidth=2)
plt.plot([vp[0],p_0[0]],[vp[1],p_0[1]],'g', linewidth=2)


#display init mesh projection

pts_2d,_,_=adjust_cylindrical_mesh(theta,h,nappe,K,vp,p_0,zmin)
display_cylindrical_mesh_proj(ax,pts_2d)


# Additional visualization elements
plt.plot([0, umax, umax, 0, 0], [0, 0, vmax, vmax, 0])
plt.axis([-0.5 * umax, 1.5 * umax, -0.5 * vmax, 1.5 * vmax])

#display frame limits

frame_limits=plt.axis()
plt.text(frame_limits[0]+0.1,frame_limits[2]+0.1,'(0,0)',fontsize=10, ha='left')


#iteractive input and adjustments

i,j,button=plt.ginput(1,show_clicks=True)[0]
while button!='f':
    if button==1:
        vp=np.array([i,j,1])  #changes vanishing point

    elif button==3:
        p_0=np.array([i,j,1])   #changes cylinder axis
    elif button=='a':
        R/=1.1 #reduce radius

    elif button=='z':
        R*=1.1 #increase radius
    elif button=='q':
        L/=1.1
        
        #remesh
        thetha,h,nappe=create_cylindrical_mesh(nb_theta,nb_h,R)

    #update mesh and recompute projection
    pts_2d,_,_=adjust_cylindrical_mesh(thetha,h,nappe,K,vp,p_0,zmin)
    display_cylindrical_mesh_proj(ax,pts_2d)

    #update visualization
    plt.plot([0,umax,umax,0,0],[0,0,vmax,vmax,0])
    plt.axis([-0.5*umax,1.5*umax,-0.5*vmax,1.5*vmax])

    #display frame limits
    frame_limits=plt.axis()
    plt.text(frame_limits[0]+0.1,frame_limits[2]+0.1,'(0,0)',fontsize=10, ha='left')


    #get user input:
    i,j,button=plt.ginput(1,show_clicks=True)[0]

#show final result:
plt.show()
    

        
    