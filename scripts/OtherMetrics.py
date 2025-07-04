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



def trendline(x,y):
    z = np.polyfit(x, y,1)
    p = np.poly1d(z)
    return p(x)



def comptime(resultsdf):
    
    num_runs = len(resultsdf)
    num_time_steps = len(resultsdf["mpc_comp_times"][0])

    average_mpc_comp_time = {ts: 0 for ts in range(num_time_steps)}

    for run in range(num_runs):
        q_mpc = resultsdf['mpc_comp_times'][run]
        for ts in range(num_time_steps):
            average_mpc_comp_time[ts] += q_mpc[ts]/num_runs

    average_mpc_comp_time_df = pd.DataFrame.from_dict(average_mpc_comp_time, orient='index')
    
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()

    time = np.linspace(900,3600,len(average_mpc_comp_time_df))
    ax.plot(time,average_mpc_comp_time_df)
    ax.plot(time, trendline(time, average_mpc_comp_time_df[0] ),"r", label = "Trend Line (Before)")

    ax.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')

    ax.set_xlabel("time step (s)", fontsize = 27)
    ax.set_ylabel("average MPC runtime", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize = 20)
    
    return fig

def BasicMetrics(resultsdf):
    num_runs = len(resultsdf)

    total_trips = 0
    completed_trips = 0
    total_travel_time = 0
    average_travel_time = 0
    total_delay = 0
    average_delay = 0

    for run in range(num_runs):
        q_met = resultsdf['df_metrics'][run]
        total_trips += q_met['total_trips']/num_runs
        completed_trips += q_met['completed_trips']/num_runs
        total_travel_time += q_met['total_travel_time']/num_runs
        average_travel_time += q_met['average_travel_time']/num_runs
        total_delay += q_met['total_delay']/num_runs
        average_delay += q_met['average_delay']/num_runs

    col_names = resultsdf["df_metrics"][0].columns.values

    data = [total_trips,completed_trips,total_travel_time,average_travel_time,total_delay,average_delay]

    data_dict = {column: dd for column, dd in zip(col_names, data)}
    index = [0]

    average_met_data_df = pd.DataFrame(data = data_dict, index = index)
    
    return average_met_data_df



def linkanalysis(resultsdf):
    num_runs = len(resultsdf)
    num_links_per_ts = len(resultsdf["df_link_summary"][0])

    arrival_count_per_link_per_ts = {link: 0 for link in range(num_links_per_ts)}
    departure_count_per_link_per_ts = {link: 0 for link in range(num_links_per_ts)}
    actual_travel_time_per_link_per_ts = {link: 0 for link in range(num_links_per_ts)}
    instantanious_travel_time_per_link_per_ts = {link: 0 for link in range(num_links_per_ts)}

    for run in range(num_runs):
        q_cl = resultsdf['df_link_summary'][run]
        for link in range(num_links_per_ts):
            arrival_count_per_link_per_ts[link] += q_cl['arrival_count'][link]/num_runs
            departure_count_per_link_per_ts[link] += q_cl['departure_count'][link]/num_runs
            actual_travel_time_per_link_per_ts[link] += q_cl['actual_travel_time'][link]/num_runs
            instantanious_travel_time_per_link_per_ts[link] += q_cl['instantanious_travel_time'][link]/num_runs

    link_ = resultsdf["df_link_summary"][0]["link"].values
    timestep = resultsdf["df_link_summary"][0]["t"].values

    col_names = resultsdf["df_link_summary"][0].columns.values

    data = [link_, timestep, arrival_count_per_link_per_ts,departure_count_per_link_per_ts,
            actual_travel_time_per_link_per_ts,instantanious_travel_time_per_link_per_ts]

    data_dict = {column: dd for column, dd in zip(col_names, data)}
    index = resultsdf["df_link_summary"][0].index.values

    average_link_data_df = pd.DataFrame(data = data_dict, index = index) 
    
    average_link_data_df = average_link_data_df[~average_link_data_df["link"].str.startswith("I")]

    instant_travel_times = {"W1I1":23.636364, "E1I1":23.636364, "N1I1": 8.181818182, "S1I1": 8.181818182,
                            "PNEIE": 7.142857143,"PNEIN": 7.142857143,"PNWIN": 7.142857143,"PNWIW": 7.142857143,
                           "PSEIE": 7.142857143,"PSEIS": 7.142857143,"PSWIS": 7.142857143,"PSWIW": 7.142857143}
    average_link_data_df['free_flow_time'] = average_link_data_df['link'].map(instant_travel_times)
    average_link_data_df["travel_delay"] = average_link_data_df["actual_travel_time"]-average_link_data_df["free_flow_time"]

    average_link_data_df_before = average_link_data_df.iloc[np.where(average_link_data_df["t"]<=900)]
    average_link_data_df_after = average_link_data_df.iloc[np.where(average_link_data_df["t"]>900)]

    average_travel_time_before = np.sum(average_link_data_df_before["actual_travel_time"])/len(average_link_data_df_before)
    average_travel_delay_before = np.sum(average_link_data_df_before["travel_delay"])/len(average_link_data_df_before)

    average_travel_time_after = np.sum(average_link_data_df_after["actual_travel_time"])/len(average_link_data_df_after)
    average_travel_delay_after = np.sum(average_link_data_df_after["travel_delay"])/len(average_link_data_df_after)

    print("BEFORE: The average travel time is ", average_travel_time_before, "s and the average delay time is", average_travel_delay_before, "s")
    print("AFTER: The average travel time is ", average_travel_time_after, "s and the average delay time is", average_travel_delay_after, "s")
    
    return [average_travel_time_before,average_travel_delay_before,average_travel_time_after,average_travel_delay_after]
    


def queue_four(resultsdf):
    
    num_runs = len(resultsdf)
    num_time_steps = len(resultsdf["df_queues"][0])

    links = resultsdf["df_queues"][0].columns.values
    num_links = len(links)
    average_queue_lengths = {link: np.zeros(num_time_steps) for link in links}

    for run in range(num_runs):
        q_df = resultsdf['df_queues'][run]
        for link in links:
            average_queue_lengths[link] += q_df[link].values

    for link in average_queue_lengths:
        average_queue_lengths[link] /= num_runs

    average_queue_per_link_per_ts_df = pd.DataFrame(average_queue_lengths)
    
    average_queue_per_link = {link: 0 for link in links}
    average_queue_per_link_before = {link: 0 for link in links}
    average_queue_per_link_after = {link: 0 for link in links}
    for link in links:
        average_queue_per_link[link] = average_queue_per_link_per_ts_df[link].sum()/num_time_steps
        average_queue_per_link_before[link] = average_queue_per_link_per_ts_df[0:30][link].sum()/num_time_steps
        average_queue_per_link_after[link] = average_queue_per_link_per_ts_df[29:][link].sum()/num_time_steps

    average_queue_per_link_df = pd.DataFrame(average_queue_per_link, index = [0])    
    
    #=== Plot Fig 1 ===
    
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()

    ax.plot(time,average_queue_per_link_per_ts_df["I1_facing_S"],label = "Raw Data")

    window_size = 3
    qq = average_queue_per_link_per_ts_df["I1_facing_S"].rolling(window=window_size).mean()
    # ax.plot(time, qq, label="Rolling Average")

    # Fit the trend line before
    ax.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df["I1_facing_S"][0:30]),"r--", label = "Trend Line (Before)",linewidth = 2)

    # after
    ax.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df["I1_facing_S"][29:]),"g--", label = "Trend Line (After)",linewidth = 2)

    ax.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')

    ax.axhline(y=average_queue_per_link_df["I1_facing_S"][0], color='grey', linestyle="-.", linewidth=2, label='Mean')


    ax.set_xlabel("time step (s)", fontsize = 27)
    ax.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize = 20)
    plt.title("I1_facing_S")
    
    #=== Plot Fig 2 ===

    fig2 = plt.figure(figsize=(20, 10))
    ax2 = plt.axes()

    ax2.plot(time,average_queue_per_link_per_ts_df["IN_to_PNE"],label = "Raw Data", marker = ",")

    window_size = 3
    qq = average_queue_per_link_per_ts_df["IN_to_PNE"].rolling(window=window_size).mean()
    # ax.plot(time, qq, label="Rolling Average")

    # Fit the trend line before
    ax2.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df["IN_to_PNE"][0:30]),"r--", label = "Trend Line (Before)")
    # after
    ax2.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df["IN_to_PNE"][29:]),"g--", label = "Trend Line (After)")


    #ax.plot(time,trendline(time, average_queue_per_link_per_ts_df["IN_to_PNE"])[0],"k--", label = "Trend Line (Total)")
    ax2.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')
    ax2.axhline(y=average_queue_per_link_df["IN_to_PNE"][0], color='grey', linestyle=":", linewidth=2, label='Mean')

    # plt.ylim([0,1])

    ax2.set_xlabel("time step (s)", fontsize = 27)
    ax2.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax2.legend(fontsize = 20)
    plt.title("IN_to_PNE")

    
    #=== Plot Fig 3 ===
    
    time = np.linspace(30,3600,num_time_steps)

    fig3 = plt.figure(figsize=(20, 10))
    ax3 = plt.axes()

    avg = average_queue_per_link_per_ts_df.iloc[:,8:12].sum(axis=1)/4
    ax3.plot(time[0:30],trendline(time[0:30],avg[0:30]), "k--",label = f"Average Link")
    ax3.plot(time[29:],trendline(time[29:],avg[29:]), "k--")

    cc = {"I1_facing_N":"r","I1_facing_S":"b","I1_facing_E":"g","I1_facing_W":"orange"}

    for link in links[8:12]:
        ax3.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df[link][0:30]), color = cc[link], label = f"{link}")
        ax3.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df[link][29:]), color = cc[link])


    ax3.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')

    ax3.set_xlabel("time step (s)", fontsize = 27)
    ax3.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax3.legend(fontsize = 20)
    
    #=== Plot Fig 4 ===
    time = np.linspace(30,3600,num_time_steps)

    fig4 = plt.figure(figsize=(20, 10))
    ax4 = plt.axes()

    avg = average_queue_per_link_per_ts_df.iloc[:,0:8].sum(axis=1)/8
    ax4.plot(time[0:30],trendline(time[0:30],avg[0:30]), "k--",label = f"Average Link")
    ax4.plot(time[29:],trendline(time[29:],avg[29:]), "k--")

    cc = {'IE_to_PNE':"r", 'IW_to_PNW':"b", 'IE_to_PSE':"g", 'IW_to_PSW':"orange", 'IN_to_PNW':"cyan",
          'IN_to_PNE':"magenta", 'IS_to_PSW':"purple", 'IS_to_PSE':"y"}

    for link in links[0:8]:
        ax4.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df[link][0:30]), color = cc[link], label = f"{link}")
        ax4.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df[link][29:]), color = cc[link])


    ax4.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')

    ax4.set_xlabel("time step (s)", fontsize = 27)
    ax4.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax4.legend(fontsize = 20)
    
    
    return fig, fig2, fig3, fig4


def queue_oneway(resultsdf):
    num_runs = len(resultsdf)
    num_time_steps = len(resultsdf["df_queues"][0])

    links = resultsdf["df_queues"][0].columns.values
    num_links = len(links)
    average_queue_lengths = {link: np.zeros(num_time_steps) for link in links}

    for run in range(num_runs):
        q_df = resultsdf['df_queues'][run]
        for link in links:
            average_queue_lengths[link] += q_df[link].values

    for link in average_queue_lengths:
        average_queue_lengths[link] /= num_runs

    average_queue_per_link_per_ts_df = pd.DataFrame(average_queue_lengths)
    
    average_queue_per_link = {link: 0 for link in links}
    average_queue_per_link_before = {link: 0 for link in links}
    average_queue_per_link_after = {link: 0 for link in links}
    for link in links:
        average_queue_per_link[link] = average_queue_per_link_per_ts_df[link].sum()/num_time_steps
        average_queue_per_link_before[link] = average_queue_per_link_per_ts_df[0:30][link].sum()/num_time_steps
        average_queue_per_link_after[link] = average_queue_per_link_per_ts_df[29:][link].sum()/num_time_steps

    average_queue_per_link_df = pd.DataFrame(average_queue_per_link, index = [0])
    
    #=== Plot Fig 1 ===
    
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()

    ax.plot(time,average_queue_per_link_per_ts_df["I1_facing_E"],label = "Raw Data", marker = ',')

    window_size = 3
    qq = average_queue_per_link_per_ts_df["I1_facing_E"].rolling(window=window_size).mean()
    # ax.plot(time, qq, label="Rolling Average")

    # Fit the trend line before
    ax.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df["I1_facing_E"][0:30]),"r--", label = "Trend Line (Before)")

    # after
    ax.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df["I1_facing_E"][29:]),"g--", label = "Trend Line (After)")

    ax.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')

    ax.axhline(y=average_queue_per_link_df["I1_facing_E"][0], color='grey', linestyle="-.", linewidth=2, label='Mean')

    ax.axhline(y=average_queue_per_link_df["I1_facing_E"][0], color='grey', linestyle="-.", linewidth=2, label='Mean')


    # plt.ylim([10,35])

    ax.set_xlabel("time step (s)", fontsize = 27)
    ax.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize = 20)
    plt.title("I1_facing_E")

    #=== Plot Fig 2 ===
    fig2 = plt.figure(figsize=(20, 10))
    ax2 = plt.axes()

    ax2.plot(time,average_queue_per_link_per_ts_df["IN_to_PNE"],label = "Raw Data")

    window_size = 3
    qq = average_queue_per_link_per_ts_df["IN_to_PNE"].rolling(window=window_size).mean()
    # ax.plot(time, qq, label="Rolling Average")

    # Fit the trend line before
    ax2.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df["IN_to_PNE"][0:30]),"r--", label = "Trend Line (Before)")

    # after
    ax2.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df["IN_to_PNE"][29:]),"g--", label = "Trend Line (After)")

    ax2.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')
    ax2.axhline(y=average_queue_per_link_df["IN_to_PNE"][0], color='grey', linestyle=":", linewidth=2, label='Mean')

    # plt.ylim([0,6])

    ax2.set_xlabel("time step (s)", fontsize = 27)
    ax2.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax2.legend(fontsize = 20)

    #=== Plot Fig 3 ===
    
    time = np.linspace(30,3600,num_time_steps)

    fig3 = plt.figure(figsize=(20, 10))
    ax3 = plt.axes()

    avg = average_queue_per_link_per_ts_df.iloc[:,8:10].sum(axis=1)/2
    ax3.plot(time[0:30],trendline(time[0:30],avg[0:30]), "k--",label = f"Average Link", linewidth=2)
    ax3.plot(time[29:],trendline(time[29:],avg[29:]), "k--", linewidth=2)

    cc = {"I1_facing_N":"r","I1_facing_S":"b","I1_facing_E":"g","I1_facing_W":"orange"}

    for link in links[8:10]:
        ax3.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df[link][0:30]), color = cc[link], label = f"{link}")
        ax3.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df[link][29:]), color = cc[link])


    ax3.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')


    ax3.set_xlabel("time step (s)", fontsize = 27)
    ax3.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax3.legend(fontsize = 20)
    
    #=== Plot Fig 4 ===
    
    time = np.linspace(30,3600,num_time_steps)

    fig4 = plt.figure(figsize=(20, 10))
    ax4 = plt.axes()

    avg = average_queue_per_link_per_ts_df.iloc[:,0:8].sum(axis=1)/8
    ax4.plot(time[0:30],trendline(time[0:30],avg[0:30]), "k--",label = f"Average", linewidth=2)
    ax4.plot(time[29:],trendline(time[29:],avg[29:]), "k--", linewidth=2)

    cc = {'IE_to_PNE':"r", 'IW_to_PNW':"b", 'IE_to_PSE':"g", 'IW_to_PSW':"orange", 'IN_to_PNW':"cyan",
          'IN_to_PNE':"magenta", 'IS_to_PSW':"purple", 'IS_to_PSE':"y"}

    for link in links[0:8]:
        ax4.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df[link][0:30]), color = cc[link], label = f"{link}")
        ax4.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df[link][29:]), color = cc[link])


    ax4.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')


    ax4.set_xlabel("time step (s)", fontsize = 27)
    ax4.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax4.legend(fontsize = 20, loc="upper right")
    
    return fig, fig2, fig3, fig4
    


def queue_three(resultsdf):
    
    #=== QUEUE LENTH ANALYSIS
    
    num_runs = len(resultsdf)
    num_time_steps = len(resultsdf["df_queues"][0])

    links = resultsdf["df_queues"][0].columns.values
    num_links = len(links)
    average_queue_lengths = {link: np.zeros(num_time_steps) for link in links}

    for run in range(num_runs):
        q_df = resultsdf['df_queues'][run]
        for link in links:
            average_queue_lengths[link] += q_df[link].values

    for link in average_queue_lengths:
        average_queue_lengths[link] /= num_runs

    average_queue_per_link_per_ts_df = pd.DataFrame(average_queue_lengths)
    
    average_queue_per_link = {link: 0 for link in links}
    for link in links:
        average_queue_per_link[link] = average_queue_per_link_per_ts_df[link].sum()/num_time_steps

    average_queue_per_link_df = pd.DataFrame(average_queue_per_link, index = [0])    
    
    #=== Plot Fig 1 ===
    
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()

    ax.plot(time,average_queue_per_link_per_ts_df["I1_facing_E"],label = "Raw Data")

    window_size = 3
    qq = average_queue_per_link_per_ts_df["I1_facing_E"].rolling(window=window_size).mean()
    # ax.plot(time, qq, label="Rolling Average")

    # Fit the trend line before
    ax.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df["I1_facing_E"][0:30]),"r--", label = "Trend Line (Before)")

    # after
    ax.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df["I1_facing_E"][29:]),"g--", label = "Trend Line (After)")

    ax.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')
    ax.axhline(y=average_queue_per_link_df["I1_facing_E"][0], color='grey', linestyle="-.", linewidth=2, label='Mean')

    ax.set_xlabel("time step (s)", fontsize = 27)
    ax.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.legend(fontsize = 20)
    plt.title("I1_facing_E")
    
    #=== Plot Fig 2 ===
    
    fig2 = plt.figure(figsize=(20, 10))
    ax2 = plt.axes()

    ax2.plot(time,average_queue_per_link_per_ts_df["IS_to_PSE"],label = "Raw Data")

    window_size = 3
    qq = average_queue_per_link_per_ts_df["IS_to_PSE"].rolling(window=window_size).mean()
    # ax.plot(time, qq, label="Rolling Average")

    # Fit the trend line before
    ax2.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df["IS_to_PSE"][0:30]),"r--", label = "Trend Line (Before)")

    # after
    ax2.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df["IS_to_PSE"][29:]),"g--", label = "Trend Line (After)")

    ax2.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')
    ax2.axhline(y=average_queue_per_link_df["IS_to_PSE"][0], color='grey', linestyle=":", linewidth=2, label='Mean')

    # plt.ylim([0,1])

    ax2.set_xlabel("time step (s)", fontsize = 27)
    ax2.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax2.legend(fontsize = 20)
    plt.title("IS_to_PSE")
    
    #=== Plot Fig 3 ===
    
    time = np.linspace(30,3600,num_time_steps)

    fig3 = plt.figure(figsize=(20, 10))
    ax3 = plt.axes()

    avg = average_queue_per_link_per_ts_df.iloc[:,6:9].sum(axis=1)/4
    ax3.plot(time[0:30],trendline(time[0:30],avg[0:30]), "k--",label = f"Average")
    ax3.plot(time[29:],trendline(time[29:],avg[29:]), "k--")

    cc = {"I1_facing_N":"r","I1_facing_S":"b","I1_facing_E":"g","I1_facing_W":"orange"}

    for link in links[6:9]:
        ax3.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df[link][0:30]), color = cc[link], label = f"{link}")
        ax3.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df[link][29:]), color = cc[link])


    ax3.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')



    ax3.set_xlabel("time step (s)", fontsize = 27)
    ax3.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax3.legend(fontsize = 20)
    
    #=== Plot Fig 4 ===
    
    time = np.linspace(30,3600,num_time_steps)

    fig4 = plt.figure(figsize=(20, 10))
    ax4 = plt.axes()

    avg = average_queue_per_link_per_ts_df.iloc[:,0:6].sum(axis=1)/8
    ax4.plot(time[0:30],trendline(time[0:30],avg[0:30]), "k--",label = f"Average")
    ax4.plot(time[29:],trendline(time[29:],avg[29:]), "k--")

    cc = {'IE_to_PNE':"r", 'IW_to_PNW':"b", 'IE_to_PSE':"g", 'IW_to_PSW':"orange", 'IN_to_PNW':"cyan",
          'IN_to_PNE':"magenta", 'IS_to_PSW':"purple", 'IS_to_PSE':"y"}

    for link in links[0:6]:
        ax4.plot(time[0:30],trendline(time[0:30], average_queue_per_link_per_ts_df[link][0:30]), color = cc[link], label = f"{link}")
        ax4.plot(time[29:],trendline(time[29:], average_queue_per_link_per_ts_df[link][29:]), color = cc[link])


    ax4.axvline(x=900, color='grey', linestyle=":", linewidth=2, label='Control Logic Start')



    ax4.set_xlabel("time step (s)", fontsize = 27)
    ax4.set_ylabel("average queue length", fontsize = 27)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax4.legend(fontsize = 20)
    
    return fig, fig2, fig3, fig4





