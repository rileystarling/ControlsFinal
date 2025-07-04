
# Author: Riley Starling
# UNI: rs4635
# Date: 14 May 2025

from uxsim import *
import random
import itertools
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict
import time


def threelink_run(iterations):
    
    results = []
    runtimes = []
    
    #--- Initiate Simulation ---
    
    for run in range(iterations):

        start_run_time = time.time()

        mpc_comp_time, df1, df2, df3, df4, df5 = threelink(run)

        end_run_time = time.time()
        total_run_time = end_run_time-start_run_time
        runtimes.append(total_run_time)

        results.append({
            "run": run,
            "mpc_comp_times": mpc_comp_time,
            "df_queues": df1,
            "df_od_summary": df2,
            "df_metrics": df3,
            "df_veh_log": df4,
            "df_link_summary":df5
        })
    resultsdf = pd.DataFrame(results)
    return resultsdf

def threelink(run):

    #--- World Generation ---

    W = World(
        name="",
        deltan=1,
        tmax=3600,
        print_mode=1, save_mode=0, show_mode=1,
        random_seed=run,
        duo_update_time=1
    )

    # network definition
    ave_len = 1.2
    st_len = 1
    cw_len = 0.4

    I1 = W.addNode("I1", 0, 0, signal=[60,60])
    W1 = W.addNode("W1", -ave_len, 0)
    E1 = W.addNode("E1", ave_len, 0)
    N1 = W.addNode("N1", 0, st_len)
    S1 = W.addNode("S1", 0, -st_len)

    IN = W.addNode("IN", 0, cw_len, signal=[60,60])
    IS = W.addNode("IS", 0, -cw_len, signal=[60,60])
    IE = W.addNode("IE", cw_len, 0, signal=[60,60])
    IW = W.addNode("IW", -cw_len, 0, signal=[60,60])

    PNW = W.addNode("PNW", -cw_len, cw_len)
    PNE = W.addNode("PNE", cw_len, cw_len)
    PSW = W.addNode("PSW", -cw_len, -cw_len)
    PSE = W.addNode("PSE", cw_len, -cw_len)

    # STREETS AND AVENUES
    #E <-> W direction: signal group 1
    # avenues
    for n1,n2 in [[W1, I1], [I1, E1]]:
        W.addLink(n1.name+n2.name, n1, n2, length=260, signal_group=1,number_of_lanes = 2,free_flow_speed = 11, jam_density_per_lane = 0.2)
        W.addLink(n2.name+n1.name, n2, n1, length=260, signal_group=1,number_of_lanes = 2,free_flow_speed = 11, jam_density_per_lane = 0.2)

    #N <-> S direction: signal group 0
    # streets
    for n1,n2 in [[I1, S1]]:
        W.addLink(n1.name+n2.name, n1, n2, length=90, signal_group=0,number_of_lanes = 2,free_flow_speed = 11, jam_density_per_lane = 0.2)
        W.addLink(n2.name+n1.name, n2, n1, length=90, signal_group=0,number_of_lanes = 2,free_flow_speed = 11, jam_density_per_lane = 0.2)

    # PEDESTRIAN CROSS WALK
    #E <-> W direction: signal group 1
    for n1,n2 in [[PSW,IS],[IS,PSE]]:
        W.addLink(n1.name+n2.name, n1, n2, length=10, signal_group=1,number_of_lanes = 3,free_flow_speed = 1.4, jam_density_per_lane = 1)
        W.addLink(n2.name+n1.name, n2, n1, length=10, signal_group=1,number_of_lanes = 3,free_flow_speed = 1.4, jam_density_per_lane = 1)

    #N <-> S direction: signal group 0
    for n1,n2 in [[PNW,IW],[IW,PSW],[PNE,IE],[IE,PSE]]:
        W.addLink(n1.name+n2.name, n1, n2, length=10, signal_group=0,number_of_lanes = 3,free_flow_speed = 1.4, jam_density_per_lane = 1)
        W.addLink(n2.name+n1.name, n2, n1, length=10, signal_group=0,number_of_lanes = 3,free_flow_speed = 1.4, jam_density_per_lane = 1)
   
    # === Demand Generation ===

    # VEHICLES
    dt = 30 
    demand_NS = 0.5 # ~15 vehicles / 30 sec
    demand_WE = 1 # ~30 vehicles / 30 sec
    entry_points = {
        W1: (demand_WE, [E1, S1], [0.7,0.3]), # straight, right
        E1: (demand_WE, [W1, S1], [0.8,0.2]), # straight, left
        S1: (demand_NS, [E1, W1], [0.625, 0.375]) # right, left
    }
    for origin, (demand, possible_dests, weights) in entry_points.items():
        for t in range(0, 3600, dt):
            dest = random.choices(possible_dests, weights = weights)[0]
            W.adddemand(origin, dest, t, t+dt, random.uniform(demand-0.3, demand))

    # PEDESTRIANS
    demand = 0.4
    entry_points = {
        PNW: (demand, [PSW]), 
        PNE: (demand, [PSE]),
        PSE: (demand, [PNE, PSW]),
        PSW: (demand, [PSE, PNW])
    }
    for origin, (demand, possible_dests) in entry_points.items():
        for t in range(0, 3600, dt):
            dest = random.choice(possible_dests)
            W.adddemand(origin, dest, t, t+dt, random.uniform(0.2, demand))

    # === Run Simulation === 

    H = 3  # Prediction horizon (in time steps)

    # NS pedestirans
    vehicles_at_IE_going_PNE = []
    vehicles_at_IW_going_PNW = []
    vehicles_at_IE_going_PSE = []
    vehicles_at_IW_going_PSW = []

    # EW pedestirans
    vehicles_at_IS_going_PSW = []
    vehicles_at_IS_going_PSE = []

    # vehicles
    vehicles_at_I1_facing_E = []
    vehicles_at_I1_facing_W = []
    vehicles_at_I1_facing_N = []

    mpc_comp_time = []
    sim_time = 0
    delay_duration = 900
    while W.check_simulation_ongoing():
        start_mpc = time.time()
        sim_time += 30
        
        # Execute the simulation in increments of 30 seconds
        W.exec_simulation(duration_t2=30)

        # PEDESTRIAN QUEUE TRACKING
        S1 = list(PNW.inlinks.values())[0]
        vehicles_at_IW_going_PNW.append(S1.num_vehicles_queue)

        S0, S1 = PSW.inlinks.values()
        vehicles_at_IS_going_PSW.append(S0.num_vehicles_queue)
        vehicles_at_IW_going_PSW.append(S1.num_vehicles_queue)

        S1 = list(PNE.inlinks.values())[0]
        vehicles_at_IE_going_PNE.append(S1.num_vehicles_queue)

        S0, S1 = PSE.inlinks.values()
        vehicles_at_IS_going_PSE.append(S0.num_vehicles_queue)
        vehicles_at_IE_going_PSE.append(S1.num_vehicles_queue)   

        # VEHICULAR QUEUE TRACKING
        LW,LE,LS = I1.inlinks.values()
        vehicles_at_I1_facing_E.append(LW.num_vehicles_queue)
        vehicles_at_I1_facing_W.append(LE.num_vehicles_queue)
        vehicles_at_I1_facing_N.append(LS.num_vehicles_queue)
        
        if sim_time > delay_duration:

            # === Main MPC Logic ===

            best_cost = float('inf')
            best_sequence = None

            for phase_seq in product([0, 1], repeat=H):

                prediction = predict(H, phase_seq, vehicles_at_I1_facing_N, 
                                     vehicles_at_I1_facing_E, 
                                     vehicles_at_I1_facing_W,
                                     vehicles_at_IW_going_PSW,
                                     vehicles_at_IW_going_PNW,
                                     vehicles_at_IE_going_PSE,
                                     vehicles_at_IE_going_PNE,
                                     vehicles_at_IS_going_PSE,
                                     vehicles_at_IS_going_PSW)

                cost = evaluate_cost(prediction,phase_seq)

                best_cost
                if cost < best_cost:
                    best_cost = cost
                    best_sequence = phase_seq

            end_mpc = time.time()
            tot_mpc_time = end_mpc-start_mpc
            mpc_comp_time.append(tot_mpc_time)

            for ii in [I1,IN,IS,IE,IW]:
                ii.signal_phase = best_sequence[0]
                ii.signal_t = 0

    # === Save Results ===

    # queue length data
    df_q = pd.DataFrame(data={'IE_to_PNE' : vehicles_at_IE_going_PNE,
                         "IW_to_PNW" : vehicles_at_IW_going_PNW,
                         "IE_to_PSE" : vehicles_at_IE_going_PSE,
                         "IW_to_PSW" : vehicles_at_IW_going_PSW,
                         "IS_to_PSW" : vehicles_at_IS_going_PSW,
                         "IS_to_PSE" : vehicles_at_IS_going_PSE,
                         "I1_facing_E" : vehicles_at_I1_facing_E,
                         "I1_facing_W" : vehicles_at_I1_facing_W,
                         "I1_facing_N" : vehicles_at_I1_facing_N
                        })

    # OD-specific traffic data
    df_od = W.analyzer.od_to_pandas()

    # simple mettrics
    df_m = W.analyzer.basic_to_pandas()

    # vehicle lof data
    df_v = W.analyzer.vehicle_trip_to_pandas()

    # link data
    df_l = W.analyzer.link_cumulative_to_pandas()

    return mpc_comp_time, df_q, df_od, df_m, df_v, df_l


def predict(H, phase_seq, vehicles_at_I1_facing_N=0, 
            vehicles_at_I1_facing_E=0, 
            vehicles_at_I1_facing_W=0,
            vehicles_at_IW_going_PSW=0,
            vehicles_at_IW_going_PNW=0,
            vehicles_at_IE_going_PSE=0,
            vehicles_at_IE_going_PNE=0,
            vehicles_at_IS_going_PSE=0,
            vehicles_at_IS_going_PSW=0
           ):
    
    # queue: current observed state
    qN = vehicles_at_I1_facing_N[-1]
    qS = 0
    qE = vehicles_at_I1_facing_E[-1]
    qW = vehicles_at_I1_facing_W[-1]
    
    # NS pedestiran queue
    qPNW_PSW = vehicles_at_IW_going_PSW[-1]
    qPSW_PNW = vehicles_at_IW_going_PNW[-1]
    qPNE_PSE = vehicles_at_IE_going_PSE[-1]
    qPSE_PNE = vehicles_at_IE_going_PNE[-1]

    # EW pedestiran queue
    qPNW_PNE = 0
    qPNE_PNW = 0
    qPSW_PSE = vehicles_at_IS_going_PSE[-1]
    qPSE_PSW = vehicles_at_IS_going_PSW[-1]

    
    # state vector
    xk = np.array([[qN],[qS],[qE],[qW], # vehicle queues
                   [qPNW_PSW],[qPSW_PNW],[qPNE_PSE],[qPNE_PSE], # N/S pedestrian queues
                   [qPNW_PNE],[qPNE_PNW],[qPSW_PSE],[qPSE_PSW]]) # E/W pedestrian queues
    
    # matricies
    
    A = np.eye(12)
    
    # NS vehicles, EW pedestrians
    B1 = np.zeros((12, 12))
    B1[0, 0] = 0  # qN
    B1[1, 1] = -0.6  # qS
    B1[8, 8] = 0  # qPNW_PNE
    B1[9, 9] = 0  # qPNE_PNW
    B1[10, 10] = -0.4  # qPSW_PSE
    B1[11, 11] = -0.4  # qPSE_PSWB2
    
    # EW vehicles, NS pedestrians
    B2 = np.zeros((12, 12))
    B2[2, 2] = -0.6  # qE
    B2[3, 3] = -0.6  # qW
    B2[4, 4] = -0.4  # qPNW_PSW
    B2[5, 5] = -0.4  # qPSW_PNW
    B2[6, 6] = -0.4  # qPNE_PSE
    B2[7, 7] = -0.4  # qPSE_PNE   
    
    B = [B1, B2]
    
    C = []
    for _ in range(3):
        Ct = np.zeros((12,1))
        Ct[1,0] = np.random.uniform(0.1, 0.5)
        for i in range(4,8):
            Ct[i, 0] = np.random.uniform(0.5, 1.2)
        Ct[10,0] = np.random.uniform(0.1, 0.5)
        Ct[11,0] = np.random.uniform(0.1, 0.5)
        C.append(Ct)

    # prediction

    predicted_queues = []
    
    for t in range(H):
        
        delta = np.zeros(2)
        delta[phase_seq[t]] = 1  # activate one phase
        
        x_next = A @ xk + sum(delta[i] * (B[i] @ xk) for i in range(2)) + C[t]
        x_next = np.maximum(x_next, 0)  # no negative queues

        predicted_queues.append(x_next.flatten())
        xk = x_next

    return predicted_queues


def evaluate_cost(predicted_queues, phase_seq, w_x=1.0, w_delta=5.0):
    cost_queues = 0
    cost_switching = 0 

    # Queue cost
    for state in predicted_queues:
        cost_queues += np.sum(state)**2

    # Switching cost
    for i in range(1, len(phase_seq)):
        if phase_seq[i] != phase_seq[i-1]:
            cost_switching += (phase_seq[i] - phase_seq[i-1])**2

    # Total cost
    return w_x * cost_queues + w_delta * cost_switching




