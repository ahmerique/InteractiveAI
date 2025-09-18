
"""
Standalone simulator made to reproduce a corruption error on step 52 on a large environment
Based on PowerGrid_poc_simulator_consol.py but without the communication module and it does not require InteractiveAI to work.
To reproduce the bug, put a breakpoint at line 118 with the condition obs.current_step>=52 at the next step it should break.
// if self._stop_if_anticipation_security_analysis(f_obs, f_env, contingency_line_ids)
"""


from grid2op.Agent import recoPowerlineAgent, AlertAgent
from config.config import logging
from grid2op.Chronics.handlers import PerfectForecastHandler, CSVHandler, DoNothingHandler
import re
import importlib
from grid2op.Agent import BaseAgent
from grid2op.Chronics import FromHandlers
import grid2op
from lightsim2grid import SecurityAnalysis
import numpy as np
import toml
import os
import time
from datetime import datetime, timedelta, timezone
import matplotlib
matplotlib.use('agg')
try:
    from lightsim2grid import LightSimBackend
    bkClass = LightSimBackend
except ImportError:
    from grid2op.Backend import PandaPowerBackend
    bkClass = PandaPowerBackend

class Listener():
    """Class containing simulator functions to stream and diagnose Grid2Op data and events."""

    def __init__(self, init_obs):
        """Initialize the Listener with initial observation."""
        self._current_issues = None
        self._anticipation = None
        self.line_statuses = init_obs.line_status
        self.subs_on_bus_2 = np.repeat(False, init_obs.n_sub)
        self.objs_on_bus_2 = {id: [] for id in range(init_obs.n_sub)}

    def _stop_if_action(self, act):
        """Check if the given action can affect the grid."""
        if act.can_affect_something():
            logging.info("The current action has a chance to change the grid")
            return True
        return False

    def _stop_if_bad_kpi(self, obs):
        """Check if there is an overload in the grid."""
        # Check if overload
        if obs.rho.max() >= 1.0:
            # logging.info("Overload")
            return True
        return False

    def _stop_if_line_disconnected(self, obs):
        """Check if any line is disconnected."""
        if np.any(obs.line_status == False):
            # logging.info("Line disconnected")
            return True
        return False

    def _stop_if_alarm(self, obs):
        """Check if an alarm is raised by the assistant."""
        do_stop_if_alarm = True
        if do_stop_if_alarm:
            if np.any(obs.time_since_last_alarm == 0):
                logging.info("Assistant raised an alarm")
                return True
        return False

    def _stop_if_anticipation_security_analysis(self, obs, env, contingency_line_ids):
        """Perform security analysis for anticipation of N-1 events."""
        # Launch the security analysis
        anticipation = []
        security_analysis = SecurityAnalysis(env)
        thermal_limit = obs.thermal_limit

        for value in (contingency_line_ids):
            security_analysis.add_single_contingency(value)

        _, res_a, _ = security_analysis.get_flows()
        for i, c_value in enumerate(contingency_line_ids):
            flow = np.array(res_a[i])
            impacted_lines = []
            rho = []
            for j, value in enumerate(flow):
                if value / thermal_limit[j] >= 1.0:
                    impacted_lines = get_formatted_name_line(obs,j)
                    rho.append(value / thermal_limit[j])
            if len(impacted_lines) > 0:
                line_name = get_formatted_name_line(obs,c_value)
                anticipation.append((line_name, impacted_lines, rho))

        self._anticipation = None
        if len(anticipation) > 0:
            self._anticipation = anticipation
            return True
        return False

    def _stop_if_issue(self, obs, f_obs, f_env, contingency_line_ids):
        """Check for various issues in the grid."""
        issues = []
        self._current_issues = []
        if self._stop_if_alarm(obs):
            issues.append("Assistant raised an alarm")

        if self._stop_if_bad_kpi(obs):
            issues.append("Overload")

        if self._stop_if_line_disconnected(obs):  # and obs.current_step == 100
            issues.append("Line lost")

        if f_obs is not None:
            # and obs.current_step == 100
            if self._stop_if_anticipation_security_analysis(f_obs, f_env, contingency_line_ids):
                issues.append("Anticipation N-1")

        if len(issues) > 0:
            self._current_issues = issues
            return True
        return False

    def stop_for_issue_state(self, obs, f_obs, f_env, contingency_line_ids):
        """Transfer private result to the simulator."""
        return self._stop_if_issue(obs, f_obs, f_env, contingency_line_ids)

    def update_objs_on_bus_switch(self, objs_on_bus_2, elem, pos_topo_vect):
        """Update objects on bus 2 after a bus switch."""
        if pos_topo_vect[elem["object_id"]] in objs_on_bus_2[elem["substation"]]:
            # elem was on bus 2,remove it from objs_on_bus_2
            objs_on_bus_2[elem["substation"]] = [
                x
                for x in objs_on_bus_2[elem["substation"]]
                if x != pos_topo_vect[elem["object_id"]]
            ]
        else:
            objs_on_bus_2[elem["substation"]].append(
                pos_topo_vect[elem["object_id"]])
        return objs_on_bus_2

    def update_objs_on_bus_assign(self, objs_on_bus_2, elem, pos_topo_vect):
        """Update objects on bus 2 after a bus assignment."""
        if (
            pos_topo_vect[elem["object_id"]
                          ] in objs_on_bus_2[elem["substation"]]
            and elem["bus"] == 1
        ):
            # elem was on bus 2,remove it from objs_on_bus_2
            objs_on_bus_2[elem["substation"]] = [
                x
                for x in objs_on_bus_2[elem["substation"]]
                if x != pos_topo_vect[elem["object_id"]]
            ]
        elif (
            pos_topo_vect[elem["object_id"]
                          ] not in objs_on_bus_2[elem["substation"]]
            and elem["bus"] == 2
        ):
            objs_on_bus_2[elem["substation"]].append(
                pos_topo_vect[elem["object_id"]])
        return objs_on_bus_2

    def update_objs_on_bus(self, objs_on_bus_2, elem, topo_vect_dict, kind):
        """Update objects on bus 2 based on topology changes."""
        for object_type, pos_topo_vect in topo_vect_dict.items():
            if elem["object_type"] == object_type and elem["bus"]:
                if kind == "bus_switch":
                    objs_on_bus_2 = self.update_objs_on_bus_switch(
                        objs_on_bus_2, elem, pos_topo_vect)
                else:
                    objs_on_bus_2 = self.update_objs_on_bus_assign(
                        objs_on_bus_2, elem, pos_topo_vect
                    )
                break
        return objs_on_bus_2

    def get_distance_from_obs(self, act, line_statuses, subs_on_bus_2, objs_on_bus_2, obs):
        """Calculate the distance from the reference topology."""
        impact_on_objs = act.impact_on_objects()

        # lines reconnetions/disconnections
        line_statuses[
            impact_on_objs["force_line"]["disconnections"]["powerlines"]
        ] = False
        line_statuses[
            impact_on_objs["force_line"]["reconnections"]["powerlines"]
        ] = True
        line_statuses[impact_on_objs["switch_line"]["powerlines"]] = np.invert(
            line_statuses[impact_on_objs["switch_line"]["powerlines"]]
        )

        topo_vect_dict = {
            "load": obs.load_pos_topo_vect,
            "generator": obs.gen_pos_topo_vect,
            "line (extremity)": obs.line_ex_pos_topo_vect,
            "line (origin)": obs.line_or_pos_topo_vect,
        }

        # Bus manipulation
        if impact_on_objs["topology"]["changed"]:
            for modif_type in ["bus_switch", "assigned_bus"]:

                for elem in impact_on_objs["topology"][modif_type]:
                    objs_on_bus_2 = self.update_objs_on_bus(
                        objs_on_bus_2, elem, topo_vect_dict, kind=modif_type
                    )

            for elem in impact_on_objs["topology"]["disconnect_bus"]:
                # Disconnected bus counts as one for the distance
                subs_on_bus_2[elem["substation"]] = True

        subs_on_bus_2 = [
            True if objs_on_2 else False for _, objs_on_2 in objs_on_bus_2.items()
        ]

        distance = len(line_statuses) - \
            line_statuses.sum() + sum(subs_on_bus_2)
        return distance, line_statuses, subs_on_bus_2, objs_on_bus_2

    def trigger_kpis(self, obs, act):
        """Calculate and return various KPIs for the current state."""
        kpis = {}
        if obs.rho.max() > 1:
            kpis["max_overload"] = float(
                np.round(np.float64(obs.rho.max()), decimals=3, out=None))
        else:
            kpis["max_overload"] = ''

        kpis["renewable_energy_share"] = float(np.round(sum(obs.gen_p[np.where((obs.gen_type == "hydro") | (
            obs.gen_type == "solar") | (obs.gen_type == "wind"))])/sum(obs.gen_p), decimals=3, out=None))
        kpis["total_consumption"] = float(
            np.round(sum(obs.load_p), decimals=3, out=None))

        distance, _, _, _ = self.get_distance_from_obs(
            act, self.line_statuses, self.subs_on_bus_2, self.objs_on_bus_2, obs)
        kpis["distance_from_reference_topology"] = float(
            np.round(np.float64(distance), decimals=3, out=None))

        kpis["curtailment_volume"] = float(
            np.round(sum(obs.curtailment_mw), decimals=3, out=None))
        kpis["redispatching_volume"] = float(np.round(max(abs(sum(obs.actual_dispatch[obs.actual_dispatch > 0])), abs(
            sum(obs.actual_dispatch[obs.actual_dispatch < 0]))), decimals=3, out=None))
        return kpis

    @property
    def current_issues(self):
        """Return the current issues detected by the Listener."""
        return self._current_issues

    @property
    def anticipation(self):
        """Return the anticipation results from security analysis."""
        return self._anticipation


def search_chronic_num_from_name(scenario_name,
                                 env):
    """Find the chronic ID from its name in the data storage base."""
    found_id = None
    # Search scenario with provided name
    for id, sp in enumerate(env.chronics_handler.real_data.subpaths):
        sp_end = os.path.basename(sp)
        if sp_end == scenario_name:
            found_id = id
    return found_id


def get_curent_lines_in_bad_KPI(obs):
    """Identify the line with the worst KPI in the grid in the following format: {line_or_to_subid}:{line_ex_to_subid}:{name_line}."""
    res = (obs.rho == obs.rho.max()).tolist().index(True)
    return get_formatted_name_line(obs, res)


def get_curent_lines_lost(obs):
    """Identify disconnected lines in the grid in the following format: {line_or_to_subid}:{line_ex_to_subid}:{name_line}."""
    res = (obs.line_status is False).tolist().index(True)
    return get_formatted_name_line(obs, res)


def get_formatted_name_line(obs, idx):
    """Format line name to {line_or_to_subid}:{line_ex_to_subid}:{name_line}"""
    return f"{obs.line_or_to_subid[idx]}:{obs.line_ex_to_subid[idx]}:{obs.name_line[idx]}"


def load_assistant(assistant_path,
                   assistant_seed,
                   env,
                   logger=None):
    """utility to load the agent"""
    # lazy loading
    assistant = None
    abs_assistant_path = os.path.abspath(assistant_path)
    submission = importlib.import_module(
        f"Ressources.XD_silly_repo.submission")
    assistant = submission.make_agent(
        env.copy(), os.path.join(abs_assistant_path, "submission"))
    if not isinstance(assistant, BaseAgent):
        msg_ = "your assistant you be a grid2op.Agent.BaseAgent"
        raise RuntimeError(msg_)
    assistant.seed(int(assistant_seed))
    return assistant


def get_nbOfTimestepSinceLastObs(obs_dict,
                                 previous_step):
    """Calculate the number of timesteps since the last observation."""
    nb_timestep = int(obs_dict.get("current_step")[0])-int(previous_step)
    return nb_timestep


def run_simulator():
    """Main function to run the PowerGrid simulator based on Grid2Op platform."""
    # Logger
    logging.getLogger().setLevel(logging.INFO)
    logging.info(" Welcome to PowerGrid Simulator based on Grid2Op platform! \n")

    try:
        # Load simulation configuration
        config = toml.load("config/CONFIG.toml")

        forecasts_horizons = [5, 10, 15, 20, 25, 30]

        # Grid2OP environment definition and loading
        # env = grid2op.make(config['env_name'],backend=bkClass())
        env = grid2op.make(config['env_name'],
                           backend=bkClass(),
                           data_feeding_kwargs={
                               "gridvalueClass": FromHandlers,
                               "gen_p_handler": CSVHandler("prod_p"),
                               "load_p_handler": CSVHandler("load_p"),
                               "gen_v_handler": DoNothingHandler("prod_v"),
                               "load_q_handler": CSVHandler("load_q"),
                               "h_forecast": forecasts_horizons,
                               "gen_p_for_handler": PerfectForecastHandler("prod_p_forecasted"),
                               "load_p_for_handler": PerfectForecastHandler("load_p_forecasted"),
                               "load_q_for_handler": PerfectForecastHandler("load_q_forecasted")})

        # Initial state
        env.seed(config['env_seed'])
        id_scenario = search_chronic_num_from_name(config['scenario_name'],
                                                   env)
        env.set_id(id_scenario)  # Scenario choice
        obs = env.reset()
        print("The scenario launched is : %s \n", env.chronics_handler.get_name())

        reward = 0
        done = False
        anticipation_compute_step = config['step_start_security_analysis']

        # Assistant definition and loading
        # Uncomment the following line if required

        if config.get("assistant_path", "") != "" :
            assistant_path = config['assistant_path']
            assistant_seed = int(config['assistant_seed'])
            local_assistant = load_assistant(assistant_path, assistant_seed, env)
        else:
            local_assistant = AlertAgent(env.action_space)

        # Added for IA Agent testing (required assistantManager.py file)
        # agentM = AgentManager()

        # Agent RecoPowerLine
        agent_reco = recoPowerlineAgent.RecoPowerlineAgent(env.action_space)
        act = agent_reco.act(obs, 0)

        logging.info("The simulation is loaded.\n")

    except Exception as e:
        print(e)
        logging.info(
            "The simulation is not load properly. The program will stop.")
        exit()

    # Listener module
    listen = Listener(obs)

    event_resolved_trigger = False
    silent_mode_msg_trigger = True
    step_counter = 0
    date = datetime.now(timezone.utc)

    while not done:
        context_date = date + timedelta(minutes=float(5))*step_counter

        # Pour corriger la valeur de act à certain pas précis en vue d'avoir notre scénario cible
        # REMOVED AS I DONT CHANGE ANYTHING TO THE SCENARIO
        # act_fixed, _ = targeted_scenario_act_fixed(env,
        #                                            obs)
        # if act_fixed is not None:
        #     act = act_fixed

        # Begining of steps : Observation updates
        obs, reward, done, info = env.step(act)  # obs,reward,done,info

        # By default
        act = env.action_space({})

        # To handle between "silent mode" and "stream simulation" (with or without InteractiveAI)
        # (The stream simulation starts at step config['scenario_first_step'])
        if obs.current_step >= config['scenario_first_step']:
            logging.info("Simulation step %s", obs.current_step)

        elif obs.current_step == config['scenario_first_step'] - 1:
            print("\n")
            logging.info(f"The simulator is now connected to InteractiveAI\n")
            silent_mode_msg_trigger = False
        else:
            if silent_mode_msg_trigger:
                logging.info(f'''Status: The scenario unfolds in silent mode.\n
                             The simulator will reconnect to InteractiveAI from the step 
                             {config['scenario_first_step']} 
                             (see configuration file) \n''')
                silent_mode_msg_trigger = False
            if obs.current_step % 50 == 0:
                print("step",
                      obs.current_step, end="",
                      flush=True)
            elif obs.current_step % 10 == 0:
                print(' ... ',
                      end="",
                      flush=True)
                # time.sleep(config['stepDuration_s']/10)


        # Forecast events checking
        obs_forecast = None
        f_env = None
        if obs.current_step == anticipation_compute_step:
            anticipation_compute_step = obs.current_step + \
                config['refresh_frequency_step']
            # pour dans 15 min (time_step_forecast=3)
            obs_forecast, *_ = obs.simulate(env.action_space(),
                                            config['time_step_forecast'])

            f_env = obs_forecast._obs_env

        if listen.stop_for_issue_state(obs,
                                       obs_forecast,
                                       f_env,
                                       [line_id for opponent in env._opponent.list_opponents for line_id in opponent._lines_ids]):
            # logging.info("An alarm is raised")

            # ----------------------------------------------------------------------------------
            if "Overload" in listen.current_issues:
                if obs.current_step >= config['scenario_first_step']:
                    logging.info("Status: there is an Overload")

                # act = display_parades_prompt(env,obs)
                if (obs.current_step < config['scenario_first_step']):
                    act = local_assistant.act(obs, 0)

                else:
                    # Récuperer les parades de InteractiveAI
                    # PASS FOR STANDALONE
                    pass
                # Added for IA Agent testing (required assistantManager.py file)
                # obs_dict = obs.to_json()
                # recommendation = agentM.recommendate(obs_dict)
                # parades = agentM.getlistOfParadeInfo()
                # act,__ = recommendation[0]

            if "Assistant raised an alarm" in listen.current_issues:
                if obs.current_step >= config['scenario_first_step']:

                    # com.push_step = obs.current_step + send_tempo
                    # if com.CAB_API_on is True and context_just_sent == False:
                    #     com.send_context_online(env,obs,config['scenario_first_step'],context_date)
                    #     context_just_sent = True

                    logging.info("Status: there is an IA Agent alert")

            if "Anticipation N-1" in listen.current_issues:
                if obs.current_step >= config['scenario_first_step']:

                    # #com.push_step = obs.current_step + send_tempo
                    # if com.CAB_API_on is True: # and context_just_sent == False:
                    #     com.send_context_online(obs_forecast._obs_env,obs_forecast,config['scenario_first_step'],context_date)
                    #     #context_just_sent = True
                    # time.sleep(40)

                    logging.info("Status: there is an Anticipation N-1 event")

                    for x in listen.anticipation:
                        logging.info(
                            f"There is a line lost anticipation event {x}")

                    obs_forecast = None
                    if obs.current_step >= config['scenario_first_step']:
                        # time.sleep(40)

                        # pause_simulation = input(
                        # "\n The simulation is on 'pause'!\n Press 'Enter' when you are ready to continue.")
                        pass

            if "Line lost" in listen.current_issues:
                if obs.current_step >= config['scenario_first_step']:
                    # com.push_step = obs.current_step + send_tempo
                    # if com.CAB_API_on == True and context_just_sent == False:
                    #     com.send_context_online(env,obs,config['scenario_first_step'],context_date)
                    #     context_just_sent = True
                    logging.info("Status: there is a Line lost %s",
                                 get_curent_lines_lost(obs))

                    event_resolved_trigger = True
                    if obs.current_step >= config['scenario_first_step']:
                        # time.sleep(40)

                        # pause_simulation = input(
                        # "\n The simulation is on 'pause'!\n Press 'Enter' when you are ready to continue.")
                        pass
            # --------------------------------------------------------------------

        # To reconnect lines in the grid any time this agent detect a line disconnection.
        # (This act is ovewriten in case of Oveload and XD_Silly intervene)
        if act == env.action_space({}):
            act = agent_reco.act(obs, 0)
            # print("Recopowerline acted. \n")

        # To handle simulator speed
        if obs.current_step >= config['scenario_first_step']:
            step_counter = step_counter + 1
            time.sleep(config['stepDuration_s'])


if __name__ == '__main__':
    run_simulator()
