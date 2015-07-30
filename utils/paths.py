from glob import glob

chi_scan_dirname_sets_1 = [
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=1,type=S,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=1,type=S,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=1,type=T,dtMem=0.1,tMem=5,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=1,type=T,dtMem=0.1,tMem=5,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=2,type=S,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=2,type=S,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=2,type=T,dtMem=0.1,tMem=5,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,Dr=1,chi=*,side=2,type=T,dtMem=0.1,tMem=5,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=1,type=S,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=1,type=S,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=1,type=T,dtMem=0.1,tMem=5,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=1,type=T,dtMem=0.1,tMem=5,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=2,type=S,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=2,type=S,noObs')],
    ['ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=2,type=T,dtMem=0.1,tMem=5,noObs', glob('ships_2D,dt=0.01,n=5000,align=0,origin=1,v=20,p=1,chi=*,side=2,type=T,dtMem=0.1,tMem=5,noObs')],
]

chi_scan_dirname_sets_2 = [
    glob('ships_1D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=1,pType=S,noObs'),
    glob('ships_1D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=1,pType=T,dtMem=0.1,ptMem=5,noObs'),
    glob('ships_1D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=2,pType=S,noObs'),
    glob('ships_1D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=2,pType=T,dtMem=0.1,ptMem=5,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,Dr=1,DChi=*,DSide=1,DType=S,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,Dr=1,DChi=*,DSide=1,DType=T,DtMem=5,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,Dr=1,DChi=*,DSide=2,DType=S,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,Dr=1,DChi=*,DSide=2,DType=T,DtMem=5,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=1,pType=S,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=1,pType=T,dtMem=0.1,ptMem=5,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=2,pType=S,noObs'),
    glob('ships_2D,dt=0.01,n=1000,align=0,origin=1,v=1,p=1,pChi=*,pSide=2,pType=T,dtMem=0.1,ptMem=5,noObs'),
]
