# Author: Riley Starling
# UNI: rs4635
# Date: 14 May 2025


import sys
sys.path.append("./scripts")

from FourLinkSimulation import fourlink_run
from OneWaySimulation import oneway_run
from ThreeLinkSimulation import threelink_run
from OtherMetrics import *

def main():
    # simulate data
    resultsdf_four = fourlink_run(15)
    fig1, fig2, fig3, fig4 = queue_four(resultsdf_four)
    fig1.savefig(f"figures/queue_four1.png")
    fig2.savefig(f"figures/queue_four2.png")
    fig3.savefig(f"figures/queue_four3.png")
    fig4.savefig(f"figures/queue_four4.png")
        
    resultsdf_oneway = oneway_run(15)
    fig1, fig2, fig3, fig4 = queue_four(resultsdf_oneway)
    fig1.savefig(f"figures/queue_oneway1.png")
    fig2.savefig(f"figures/queue_oneway2.png")
    fig3.savefig(f"figures/queue_oneway3.png")
    fig4.savefig(f"figures/queue_oneway4.png")
    
    resultsdf_three = threelink_run(15)
    fig1, fig2, fig3, fig4 = queue_three(resultsdf_three)
    fig1.savefig(f"figures/queue_three1.png")
    fig2.savefig(f"figures/queue_three2.png")
    fig3.savefig(f"figures/queue_three3.png")
    fig4.savefig(f"figures/queue_three4.png")
    
    
    rrs = [resultsdf_four,resultsdf_oneway,resultsdf_three]
    
    for rr in rr:
        
        # Do computation time
        fig = comptime(rr)
        fig.savefig(f"figures/mpc_runtime_{rr}.png")

        # Do computation time
        linkanalysis(rr) # prints result

        M = BasicMetrics(rr) # metrics dataframe
    
        return M

    print("All plots generated and saved.")

if __name__ == "__main__":
    main()

