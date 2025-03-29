import lhapdf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Q2=10

xinit=1.000000e-04
deltax=1e-3
xdata=np.arange(xinit, 1, deltax)

zinit = 1.000000e-04
deltaz = 1e-3
zdata = np.arange(zinit, 1, deltaz)


PDFdataset = lhapdf.mkPDF("cteq61")
FF_pion_dataset=["NNFF10_PIp_nlo"]
FF_kaon_dataset=["NNFF10_KAp_nlo"]

def xFxQ2(dataset,flavor,x,QQ):
    temp_parton_dist_x=[(dataset.xfxQ2(flavor, x[i], QQ)) for i in range(len(x))]
    return temp_parton_dist_x

def zFzQ(dataset,flavor,zz,QQ):
    # Here "0" represents the central values from the girds
    temp_zD1=lhapdf.mkPDF(dataset[0], 0)
    zD1_vec=[(temp_zD1.xfxQ2(flavor,zz[i],QQ)) for i in range(len(zz))]
    return zD1_vec

f1 = plt.figure(1)
yyu=np.array(xFxQ2(PDFdataset,2,xdata,Q2))
plt.plot(xdata,yyu,color='blue')
plt.xlim(0.0001,1)
plt.ylim(0,1.0)
plt.xscale('log')

f2 = plt.figure(2)
test_FF=np.array(zFzQ(FF_pion_dataset,2,zdata,Q2))
print(test_FF)
plt.plot(zdata,test_FF,color='green')
plt.xlim(0.2,1.0)
plt.ylim(0.0,1.0)
f1.savefig("UQuark_X.pdf")
f2.savefig("UQuark_Z.pdf")





