import numpy as np
import numpy.matlib as M
import csv
import matplotlib.pyplot as plt

class var_data:
    "A class for data series used in VARs."

    def __init__(self, series, names=None, freq="generic", start=1):
        self.series = series
        (self.n, self.ny) = series.shape
        self.freq = freq
        # initialize names
        if names is None:
            self.names = dict([("Series " + str(i+1), i)
                               for i in range(self.ny)])
        elif len(names) == self.ny:
            self.names = dict([(names[i], i) for i in range(self.ny)])
        else:
            pass
        
        if freq=="generic":
            self._ind = str(range(1, self.n+1, 1))
        elif freq=="quarterly":
            if not(isinstance(start, str)):
                print "ERROR: Start date should be string for quarterly data: e.g., 1984q1"

            y0 = int(start[0:start.find("q")])
            q0 = int(start[start.find("q")+1:])
            self._ind = []
            self._ind.append(start)

            for i in range(1, self.n):
                ti = y0 + 0.25*(q0+i-1) 
                yi = str(int(ti))
                qi = str(int((ti-int(ti))/0.25+1))
                self._ind.append(yi+"q"+qi)

    @classmethod
    def empty(cls, ny=3, n=200):
        obj = var_data(M.zeros((n, ny)))
        return obj

    @classmethod
    def read_from_csv(cls, filename, header=True, freq="generic", start=1):
        """
        Reads data from a file and return a var_data object.
        filename --
        """
        test = list(csv.reader(open(filename, "rU")))

        if header:
            names = test[0]
            series = test[1:]
        else:
            names = None
            series = test

        series = map(lambda x: map(float, x), series)
        obj = var_data(np.mat(series), names, freq, start)
        return obj

    def mlag(self, p, cons=True):
        n = self.n
        ny = self.ny

        if p == 0:
            return np.mat(np.ones((n, 1)))
        
        x = M.zeros((n, ny*p))

        x[0:p, :] = None
        x[p, :] = np.reshape(self.series[0:p, :], (1, ny*p))

        for i in range(p+1, n):
            x[i, :(ny * (p - 1))] = x[i-1, ny:]
            x[i, (ny * (p - 1)):] = self.series[i-1, :]

        if cons:
            x = np.hstack((x, np.mat(np.ones((n, 1)))))
        
        return x

    def plot(self):
        plt.figure(1)
        for i in range(0, 3):
            plt.subplot(3, 1, i + 1)
            plt.plot(self.series[:, i])
            plt.title("Test")

        plt.show()
        plt.close()

    def get_series(self, t0=None, t1=None):
        """
        Returns data from date t0 (default 1) to t1 (default T).
        """
        if t0 == t1 == None:
            return self.series

        if isinstance(t0, str):
            t0 = [i for i, x in enumerate(self._ind) if x == t0]
            t0 = t0[0]

        if t1==None:
            t1 = self.series.shape[0]
        elif isinstance(t0, str):
            t1 = [i for i, x in enumerate(self._ind) if x == t1]
            t1 = t1[0]
        elif isinstance(t0, int):
            t1 = t1

        return self.series[t0:t1, :]
                
    def get_ind(self, t0):
        
        if isinstance(t0, str):
            t0 = [i for i, x in enumerate(self._ind) if x == t0]
            t0 = t0[0]
        return t0
            
            
    def __str__(self):
        print "%6s "*self.ny % tuple(self.names.keys())
        return ("% 7.2f"*self.ny + "\n")*self.n % tuple(self.series.flatten().tolist()[0])

    #def __getitem__(self, selection, series=slice(0, self.ny)):
    #    print self.series[selection, series]
        
class real_time_dataset:

    def __init__(self, originalData, estart=50, t0=0, fixed_window=True):

        print "Constructing new 'real-time' dataset using a fixed VAR time series."
        self.rt_data = []
        self.fcst_data = []
        j = originalData.get_ind(t0)
        for i in range(originalData.get_ind(estart), originalData.n-1):
  
            tmpVARData = var_data(originalData.get_series(j, i),
                                  originalData.names.keys(),
                                  freq=originalData.freq,
                                  start=originalData._ind[j])

            tmpFORData = var_data(originalData.get_series(i+1),
                                  originalData.names.keys(),
                                  freq=originalData.freq,
                                  start=originalData._ind[j+1])

            print "Constructing data set %2i: %6s - %6s" % (len(self.rt_data)+1, tmpVARData._ind[0], tmpVARData._ind[tmpVARData.n-1]),
            print "      (%2i possible forecast evaluations.)" % tmpFORData.n
            self.rt_data.append(tmpVARData)
            self.fcst_data.append(tmpFORData)
            j += 1*fixed_window
            
        print "A total of %i datasets were created." % len(self.rt_data)
        ny = tmpVARData.ny
    def size(self):
        return len(self.rt_data)

    
    def getVARData(self, i):
        return self.rt_data[i]

    def getFORData(self, i):
        return self.fcst_data[i]

    
if __name__ == "__main__":
    x = var_data.empty()
    x = var_data.empty(ny=5)
    x = var_data(M.zeros((1, 2)), ["GDP", "INF"])
    x = var_data.read_from_csv(filename="3eqvar.csv", header=True, freq="quarterly", start="1965q2")
    real_time_dataset(x, estart="1996q1", t0="1983q1")
