######################################################################
#                                                                    #
#  ParticleObservables.py                                            #
#  Author: Jenna Chisholm                                            #
#  Updated: Oct.8/25                                                 #
#                                                                    #
#  Defines observables and particles that can be used in plotting.   #
#                                                                    #
######################################################################

from DataStructures import Observable,Particle
   
# Observables (remove ticks as an option here? put that in plotting I think, also units?)
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

# Particles (remove observables?)
TH = Particle("th","t,had",[PT,PX,PY,ETA,Y,PHI,M,E,POUT],["thad","topHad","top_had"])
TL = Particle("tl","t,lep",[PT,PX,PY,ETA,Y,PHI,M,E,POUT],["tlep","topLep","top_lep"])
TTBAR = Particle("ttbar","t\\overline{t}",[PT,PX,PY,ETA,Y,PHI,M,E,DPHI,DETA,HT,YBOOST,YSTAR,CHI]) # no alterative names

PARTICLES = {particle.name: particle for particle in [TH,TL,TTBAR]}
