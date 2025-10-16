######################################################################
#                                                                    #
#  Particles_and_Observables.py                                      #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.16/25                                                #
#                                                                    #
#  Defines observables and particles that can be used in plotting.   #
#                                                                    #
######################################################################


class Observable:
    """
    Observable object class (for plotting purposes). It is a component of a variable object, and also linked to particle objects.
    """

    def __init__(self, name, label,  units='', res='Resolution', alt_names=[]):
        """
        Initializes a observable object.

            Parameters:
                name (str): Name of the observable (e.g.'pt')
                label (str): Label for the observable (note: this may be different than the name if you want to, for example, use Latex formatting).

            Options:
                units (str): Units for the observable, for axis title purposes (e.g.'GeV', default:'').
                res (str): Specifies whether this observable should use 'Resolution' or 'Residuals' (default:'Resolution').
                alt_names (list of str): Alternate names for the observable (default: []).

            Attributes:
                name (str): Name of the observable (e.g.'pt')
                label (str): Label for the observable (note: this may be different than the name if you want to, for example, use Latex formatting).
                units (str): Units for the observable (e.g.'GeV', default:'').
                units_label (str): Units for the observable, for axis title purposes (e.g.'[GeV]', default:'').
                res (str): Specifies whether this observable should use 'Resolution' or 'Residuals' (default:'Resolution').
                alt_names (list of str): Alternate names for the observable (default: []).
        """

        self.name = name
        self.label = label
        self.units = units
        self.units_label = units if units=='' else '['+units+']'
        self.res = res.capitalize()
        self.alt_names = alt_names
        
        
class Particle:
    """ 
    Particle object class (for plotting purposes). It is a component of a variable object.
    """
    
    def __init__(self, name, label, observables, alt_names=[]):
        """
        Initializes a particle object. 

            Parameters:
                name (str): Name of the particle (e.g.'th')
                label (str): Label for the particle (note: this may be different than the name if you want to, for example, use Latex formatting).
                observables (list of observable objects): List of observable objects that one could plot for this particle.

            Options:
                alt_names (list of str): Alternate names for the particle (default: []).

            Attributes:
                name (str): Name of the particle (e.g.'th').
                label (str): Label for the particle (note: this may be different than the name if you want to, for example, use Latex formatting).
                potential_observables (list of Observable objects): List of observable objects that one could potentially plot for this particle.
                alt_names (list of str): Alternate names for the particle (default: []).
        """

        self.name = name
        self.label = label
        self.potential_observables = {observable.name: observable for observable in observables}
        self.alt_names = alt_names
        
    def get_observable(self,name):
        
        return self.potential_observables[name]
  
  
#                                               #
##                                             ## 
### --------------- CONSTANTS --------------- ###
##                                             ##
#                                               #
   
# Observables (with desired units)
PT = Observable("pt","p_T","GeV","Resolution")
PX = Observable("px","p_x","GeV","Resolution")
PY = Observable("py","p_y","GeV","Resolution")
ETA = Observable("eta","\\eta","","Residuals")
Y = Observable("y","y","","Residuals")
PHI = Observable("phi","\\phi","","Residuals")
M = Observable("m","m","GeV","Residuals")
E = Observable("E","E","GeV","Residuals")
POUT = Observable("pout","p_{out}","GeV","Residuals")
DPHI = Observable("dphi","|\\Delta\\phi|","","Residuals",["deltaPhi"])
DETA = Observable("deta","|\\Delta\\eta|","","Residuals",["deltaEta"])
HT = Observable("Ht","H_T","GeV","Resolution",["HT"])
YBOOST = Observable("yboost","y_{boost}","","Residuals",["y_boost"])
YSTAR = Observable("ystar","y_{star}","","Residuals",["y_star"])
CHI = Observable("chi","\\chi","","Resolution",["chi_tt"])

# Particles
TH = Particle("th","t,had",[PT,PX,PY,ETA,Y,PHI,M,E,POUT],["thad","topHad","top_had"])
TL = Particle("tl","t,lep",[PT,PX,PY,ETA,Y,PHI,M,E,POUT],["tlep","topLep","top_lep"])
TTBAR = Particle("ttbar","t\\overline{t}",[PT,PX,PY,ETA,Y,PHI,M,E,DPHI,DETA,HT,YBOOST,YSTAR,CHI]) # no alterative names

PARTICLES = {particle.name: particle for particle in [TH,TL,TTBAR]}
