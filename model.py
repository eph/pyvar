# Economic Model Class
class Model:
    "A class for economic models."
    
    def __init__(self, lk, pr):
        # Initializations...
        print "Initializing Model"
        
    @staticmethod
    def lnpos(para):
        return lk.lnpos(para) + pr.lnpos(para)

   
