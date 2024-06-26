# Py4SciComp
Python for Scientific Computing (FEniCS, PyTorch, VTK) 


Py4SciComp is an educational effort that attempts to make three useful open-source Python-based resources in scientific computing accessible with tutorials, sample codes/data, and short theoretical lectures:

**FEniCS**: FEniCS is an open-source finite-element method (FEM) solver for solving PDEs. Users are expected to be familiar with weak form formulation of PDEs and FEniCS provides a flexible environment for coding the weak forms and the FEM problem. Behind the curtains, FEniCS converts the Python code to an HPC-supported C++ code and runs it. \
**PyTorch**: PyTorch is a very popular framework for implementing deep learning, automatic differentiation, and optimization based codes. \
**VTK**: Visualization Toolkit (VTK) libraries provide various useful tools for efficient pre- and post- processing of mesh-based and discrete data. Popular visualization software like ParaView is built based on VTK.\
**A Unified Python cyberinfrastructure**: It is the hope that these resources motivate development of unified Python cyberinfrastructure that integrates these resources for scientific computing. 


In this GitHub directory, we provide a series of tutorials. Each tutorial is accompanied by a short Youtube video, sample codes, and data. \
The Youtube videos should be viewed in HD mode for clarity.


Available resources (**to be expanded in the future**): 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FEniCS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**FEniCS**: \
Youtube playlist:\
https://www.youtube.com/playlist?list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD

1-Introduction \
Youtube tutorial: \
https://www.youtube.com/watch?v=P642Oq-TohY&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=1&t=163s&ab_channel=AmirhosseinArzani

2-Automated parametric solution \
Youtube tutorial: \
https://www.youtube.com/watch?v=I9UJNsStOo4&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=2&t=437s&ab_channel=AmirhosseinArzani

3-Steady Navier-Stokes \
Youtube tutorial: \
https://www.youtube.com/watch?v=4sITKq0e6Mo&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=3&t=1s&ab_channel=AmirhosseinArzani

4-Unsteady Navier-Stokes \
Youtube tutorial: \
https://www.youtube.com/watch?v=pmjVsZU3jOE&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=4&ab_channel=AmirhosseinArzani

5-2D mass transport (advection-diffusion) \
Youtube tutorial: \
https://www.youtube.com/watch?v=Qpk_4oK01zk&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=5&ab_channel=AmirhosseinArzani

6-Importing 3D mesh into FeniCS \
Youtube tutorial (part I): \
https://www.youtube.com/watch?v=W40_SKjmh7w&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=6&ab_channel=AmirhosseinArzani

Youtube tutorial (part II): \
https://www.youtube.com/watch?v=yp93zXRJG3E&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=7&ab_channel=AmirhosseinArzani

7- 3D mass transport in complex geometry (advection-diffusion) \
Youtube tutorial: \
https://www.youtube.com/watch?v=aaVY2yWGPTs&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=8&ab_channel=AmirhosseinArzani

8- 3D Biotransport (advection-diffusion) \
Codes for this tutorial are available at: https://github.com/amir-cardiolab/Biotransport_FEniCS \
Youtube tutorial: \
https://www.youtube.com/watch?v=iMob-dfDSUs&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=9&ab_channel=AmirhosseinArzani

9- Residence-time (flow stagnation) \
Youtube tutorial: \
https://www.youtube.com/watch?v=QOq8rYziqCI&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=10&ab_channel=AmirhosseinArzani

10- Oasis: A robust package for minimally dissipative solution of Navier-Stokes \
Youtube tutorial: \
https://www.youtube.com/watch?v=mg1y5XcHwJM&list=PLw74xLHy0_j8umWkSsT-9U0CmoRBpz4vD&index=11&ab_channel=AmirhosseinArzani



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PyTorch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**PyTorch**: \
Youtube playlist: \
https://www.youtube.com/playlist?list=PLw74xLHy0_j_ROpJ--S-D9Op8_uL7S1EF

1- Data-driven neural network (2D field to 2D field mapping) \
Youtube tutorial: \
https://www.youtube.com/watch?v=CYOexv1Sg40&list=PLw74xLHy0_j_ROpJ--S-D9Op8_uL7S1EF&index=1&t=4s&ab_channel=AmirhosseinArzani

2- Physics-informed neural network (PINN) for forward problems \
Youtube tutorial: \
https://www.youtube.com/watch?v=whXM-w7ig-I&list=PLw74xLHy0_j_ROpJ--S-D9Op8_uL7S1EF&index=2&ab_channel=AmirhosseinArzani

3- PINN for inverse modeling \
Codes for this tutorial are available at: https://github.com/amir-cardiolab/PINN-wss \
Youtube tutorial: \
https://www.youtube.com/watch?v=UaJmVW8Zbew&list=PLw74xLHy0_j_ROpJ--S-D9Op8_uL7S1EF&index=3&ab_channel=AmirhosseinArzani

4- PINN for multi-fidelity and multi-physics modeling \
Codes for this tutorial are available at: https://github.com/amir-cardiolab/PINN_multiphysics_multifidelity  \
Youtube tutorial: \
https://www.youtube.com/watch?v=IKaTnZ_xhfw&list=PLw74xLHy0_j_ROpJ--S-D9Op8_uL7S1EF&index=4&ab_channel=AmirhosseinArzani

5- Pytorch output converted to VTK and Matlab \
Youtube tutorial: \
https://www.youtube.com/watch?v=nP9T04VfjeI&list=PLw74xLHy0_j_ROpJ--S-D9Op8_uL7S1EF&index=5&ab_channel=AmirhosseinArzani




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VTK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**VTK coding and ParaView**: 

Youtube playlist: \
https://www.youtube.com/playlist?list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP

1- Introduction to Python VTK coding \
Youtube tutorial: \
https://www.youtube.com/watch?v=QDJgbSQnhjc&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=1&t=1s&ab_channel=AmirhosseinArzani

2- Vector data (curl and gradient) in VTK \
Youtube tutorial: \
https://www.youtube.com/watch?v=V4eklopOLP8&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=2&ab_channel=AmirhosseinArzani

3- Interpolate between unstructured mesh data \
Youtube tutorial: \
https://www.youtube.com/watch?v=-fA02RGlmQU&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=3&ab_channel=AmirhosseinArzani

4- Wall shear stress (WSS) divergence calculation \
Youtube tutorial: \
https://www.youtube.com/watch?v=jwp7N8lLLKA&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=4&ab_channel=AmirhosseinArzani

5- Processing particle data in VTK (for Lagrangian modeling) \
Youtube tutorial: \
https://www.youtube.com/watch?v=vd8CK2NGDSM&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=5&ab_channel=AmirhosseinArzani

6- Introduction to ParaView and several useful filters \
Youtube tutorial: \
https://www.youtube.com/watch?v=WsZKrg9ABG4&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=6&ab_channel=AmirhosseinArzani

Data available here: 
https://drive.google.com/drive/u/1/folders/1edkqHqn5eKmK6RAhnk7Nj_YxDpaCakQ2

7- Advanced vector field visualization in ParaView \
Youtube tutorial: \
https://www.youtube.com/watch?v=fV5jGfqNIvk&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=7&ab_channel=AmirhosseinArzani 

Data available here: 
https://drive.google.com/drive/folders/12Oxc55K27yVMLDToE1jUhtMDRsvog69s?usp=sharing

8- ParaView's calculator for quick data post-processing \
Youtube tutorial: \
https://www.youtube.com/watch?v=r83hP3ZLr_k&list=PLw74xLHy0_j98ZWgbzlq6ZeOhND-6ceFP&index=8&ab_channel=AmirhosseinArzani



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Theory lectures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

**Lectures (theory)**: Theoretical lectures that support Py4SciComp tutorials.  

**FEM-based CFD**: \
Lectures for computational fluid dynamics (CFD) with finite-element method (FEM):

These lectures support FEniCS and are designed to provide the theoretical foundation needed for using FEniCS:

https://www.youtube.com/playlist?list=PLw74xLHy0_j_Jz0YCPoYfajhq2d9byqP0



**Scientific machine learning**: \
Lectures for scientific machine learning (machine learning for physics-based modeling).

To be added in the future. 















