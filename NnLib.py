# *********************************************
# *       A Neural Network Library            *
# * Three layer Feed Forward Neural Network   *
# * using Rumelhart's Back Propagation        *
# *                  by                       *
# *	         Himanshu Mazumdar                *
# *	 Tested with XOR DATA input output        *
# *	 Date:- 7-August-2022                     *
# *********************************************
import math
import random
from random import randint

class NNijk:
    #*****************************************
    ii = 2      # max input
    jj = 2      # max hidden
    kk = 1      # max output
    smax = 1000 # epoch training loop
    out  = []   #[KK]          Expected Output
    wkj  = []   #[KK] [JJ+1]   Weights
    wki  = []   #[KK] [II+1]   Weights
    wji  = []   #[JJ] [II+1]   Weights 
    xk   = []   #[KK]          Input to output Neuron
    yk   = []   #[KK]  		   Output of output Neuron
    xj   = []   #[JJ]  		   Input to hidden Neuron
    yj   = []   #[JJ+1]  	   Output of hidden Neuron
    yi   = []   #[II+1]  	   Output of Input Neuron
    dt   = []   #[sno][IO][sz] 0:Input,1:Output float data list
    eta  = 0.1 # learning rate
    interr = 0
    count = 0
    ioflag = 0 # 0:XOR, 1:from dt[][][]
    #*****************************************
    def __init__(self, ii, jj, kk):
        self.init_weights(ii, jj, kk)
    def init_weights(self, ii, jj, kk):
        self.ii = ii
        self.jj = jj
        self.kk = kk
        self.out = []
        self.yk = []
        self.xk = []
        self.yj = []
        self.xj = []
        self.yi = []
        self.wji = []
        self.wkj = []
        self.wki = []
        for k in range(kk):
            self.out.append(0)
            self.yk.append(0)
            self.xk.append(0)
            w1 = []
            for j in range(jj+1):
                w1.append(0)
            self.wkj.append(w1)
            w2 = []
            for i in range(ii+1):
                w2.append(0)
            self.wki.append(w2)
        for j in range(jj):
            self.yj.append(0)
            self.xj.append(0)
            w3 = []
            for i in range(ii+1):
                w3.append(0)
            self.wji.append(w3)
        self.yj.append(0)    
        for i in range(ii+1):
            self.yi.append(0)
        self.yi[self.ii] = 1;
        self.yj[self.jj] = 1;

    #****************** LOAD Random Weight   ******************/
    def load_random_weight(self):
        for k in range(self.kk): 
            for j in range(self.jj+1): 
                self.wkj[k][j] = 0.5-random.random()
        for k in range(self.kk): 
            for i in range(self.ii+1): 
                self.wki[k][i] = 0.5-random.random()
        for j in range(self.jj): 
            for i in range(self.ii+1): 
                self.wji[j][i] = 0.5-random.random()
        self.yi[self.ii]=1.0;
        self.yj[self.jj]=1.0;
    #****************** Load Input Output  ******************/
    def load_xor_to_io(self):
        # k2 = int(self.ii / 2)
        for n in range(self.ii):
            self.yi[n] = randint(0, 1)
        for n in range(self.kk):
            self.out[n] =  int(self.yi[n * 2]) ^ int(self.yi[n * 2 + 1])
    #****************** Load Input Output  ******************/
    def load_file_dt_to_io(self):
        sz = len(self.dt)
        if sz == 0:
            print('data not loaded')
            return
        no = random.randint(0, sz-1)
        for n in range(self.ii):
            self.yi[n] = self.dt[no][0][n]
        for n in range(self.kk):
            self.out[n] =  self.dt[no][1][n]
    #********************  Update all Layers *********************
    def update_all_layers(self):
        for j in range(self.jj): 
            self.xj[j] = 0.0
            for i in range(self.ii + 1):
                self.xj[j] +=  self.wji[j][i] * self.yi[i]
            if self.xj[j] > 10: 
                self.xj[j] = 10			
            if self.xj[j] < -10: 
                self.xj[j] = -10
            self.yj[j] = 1.0 / ( 1.0 + math.exp(-self.xj[j]))
            a=self.yj[j]
        for k in range(self.kk):
            self.xk[k] = 0.0
            for j in range(self.jj + 1):
                self.xk[k] +=  self.wkj[k][j] * self.yj[j]
        for k in range(self.kk):
            for i in range(self.ii + 1):
                self.xk[k] +=  self.wki[k][i] * self.yi[i]
            if self.xk[k]>10: 
                self.xk[k]=10			
            if self.xk[k]<-10: 
                self.xk[k]=-10
            self.yk[k] = 1.0 / ( 1.0 + math.exp(-self.xk[k]))
        err = 0
        self.interr = 0
        for k in range(self.kk):
            p = 0
            err += math.fabs(self.yk[k]-self.out[k])
            if self.yk[k] > 0.5: 
                p = 1 
            else: 
                p = 0
            if p > self.out[k]:
                self.interr += p-self.out[k]
            else:
                self.interr += -p+self.out[k]
        return err
    #***************** Correct All Errors ****************************
    def update_and_correct_error(self):
        for k in range(self.kk):
            for j in range(self.jj + 1):
                self.wkj[k][j] -= self.eta * (self.yk[k]-self.out[k]) * self.yk[k] * (1-self.yk[k]) * self.yj[j]
        for k in range(self.kk):        
            for i in range(self.ii + 1):
                self.wki[k][i] -= self.eta * (self.yk[k]-self.out[k]) * self.yk[k] * (1-self.yk[k]) * self.yi[i]
        for j in range(self.jj):
            r=0
            for k in range(self.kk):
                r += (self.yk[k]-self.out[k]) * self.yk[k] * (1-self.yk[k]) * self.wkj[k][j]
            r = r * self.eta * self.yj[j] * (1-self.yj[j])
            for i in range(self.ii + 1):
                self.wji[j][i] -= r * self.yi[i]
                # aa=self.wji[j][i]
    #****************** Train Net  ***********************************
    def train_net(self):
        smax = 1000
        totalerr = 0.0
        totalinterr = 0
        for s in range(smax):
            if self.ioflag == 0:
                self.load_xor_to_io()
            if self.ioflag == 1:
                self.load_file_dt_to_io()
            totalerr += self.update_all_layers()
            self.update_and_correct_error()
            totalinterr += self.interr
        # print(str(self.count) +'\t'+ str(totalinterr) + '\t' + str(round(totalerr,4)))
        result = str(self.count) +','+ str(totalinterr/(self.kk*smax)) + ',' + str(round(totalerr/(self.kk*smax),4))
        self.count += 1
        return result 
    #****************** Load weights from File ***********************
    def load_weights(self, flnm):
        file1 = open(flnm,"r")
        pg0 = file1.readline().removesuffix('\n')
        wrd = file1.readline().removesuffix('\n') .split(',')
        self.ii = int(wrd[0])
        self.jj = int(wrd[1])
        self.kk = int(wrd[2])
        self.init_weights(self.ii, self.jj, self.kk)
        for j in range(self.jj):
            wsji = file1.readline().removesuffix('\n') .split(',')
            for i in range(self.ii + 1):
                self.wji[j][i]=float(wsji[i]) 
        for k in range(self.kk):
            wski = file1.readline().removesuffix('\n') .split(',')
            for i in range(self.ii + 1):
                self.wki[k][i]=float(wski[i]) 
        for k in range(self.kk):
            wskj = file1.readline().removesuffix('\n') .split(',')
            for j in range(self.jj + 1):
                self.wkj[k][j]=float(wskj[j]) 
        file1.close()
        self.yi[self.ii] = 1;
        self.yj[self.jj] = 1;
        a=123
    #****************** Save weights to File *************************
    def save_weights(self, flnm):
        file = open(flnm,"w")
        pag = "NN HSM\n"
        pag += str(self.ii) + "," + str(self.jj) + "," + str(self.kk) + "\n"
        for j in range(self.jj):
            for i in range(self.ii+1):
                pag += str(round(self.wji[j][i],4)) + ","
            pag = pag.removesuffix(',') 
            pag += "\n" 
        # pag += "\n" 
        for k in range(self.kk):
            for i in range(self.ii+1):
                pag += str(round(self.wki[k][i],4)) + ","
            pag = pag.removesuffix(',') 
            pag += "\n" 
        # pag += "\n" 
        for k in range(self.kk):
            for j in range(self.jj+1):
                pag += str(round(self.wkj[k][j],4)) + ","
            pag = pag.removesuffix(',') + "," 
            pag += "\n" 
        file.writelines(pag)
        file.close()
    #****************** Load weights from File ***********************
    def load_data(self, flnm):
        file1 = open(flnm,"r")
        lns = []
        while True:
            dts = file1.readline()
            if dts == '':
                break
            lns.append(dts)
        file1.close()
        nn = len(lns)
        w2 = lns[0].split(':')
        wi=w2[0].split(',')
        wo=w2[1].split(',')
        ii = len(wi)
        kk = len(wo)
        self.dt = []
        for n in range(nn):
            dt2 =[]
            dt2.append(0)
            dt2.append(0)
            self.dt.append(dt2)
            self.dt[n][0]=[]
            self.dt[n][1]=[]
            for i in range(ii):
                self.dt[n][0].append(0.0)
            for k in range(kk):
                self.dt[n][1].append(0.0)
        for n in range(nn):
            dts = lns[n].removesuffix('\n') .split(':')
            dts1 = dts[0].split(',')
            dts2 = dts[1].split(',')
            for i in range(ii):
                self.dt[n][0][i] = int(dts1[i])
            for k in range(kk):
                self.dt[n][1][k] = int(dts2[k])
        a = self.dt
    #*****************************************************************
