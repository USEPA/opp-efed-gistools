import os
import pandas as pd
import numpy as np
import datetime as dt


# import matplotlib.dates as pltdt
# import matplotlib.pyplot as plt

class PlantEndpoints(object):
    def __init__(self, type="", level="", plant=0, effect=0, indicator="", val=0, mrid=""):
        self.type = type
        self.level = level
        self.plant = plant
        self.effect = effect
        self.indicator = indicator
        self.val = val
        self.mrid = mrid


class PWCInputs(object):
    def __init__(self, chem_name="", soil_t12=1e8, foliar_t12=1e8, washoff=0.5, Kd=0, num_apps=1, app_days=[],
                 app_months=[],
                 app_rate=[], scenario="", app_method="Ground_Very_Fine_to_Fine_Low_Boom_90th"):
        self.chem_name = chem_name
        self.soil_t12 = soil_t12
        self.foliar_t12 = foliar_t12
        self.washoff = washoff
        self.Kd = Kd
        self.num_apps = num_apps
        self.app_days = app_days
        self.app_months = app_months
        self.app_rates = app_rate
        self.scenario = scenario
        self.app_method = app_method


class SemiPWCInputs(object):
    def __init__(self, depth=0.15, volume_sed=1500, bulk_density=0, porosity=0.5):
        self.depth = depth
        self.volume_sed = volume_sed
        self.bulk_density = bulk_density
        self.porosity = porosity
        self.volume_water = volume_sed * porosity


class hydro_inputs:
    def __init__(self, max_hold_cap=0, min_hold_cap=0, avail_cap=0, bulk_density=0, soil_mass=0):
        self.max_hold_cap = max_hold_cap
        self.min_hold_cap = min_hold_cap
        self.avail_cap = avail_cap
        self.bulk_density = bulk_density
        self.soil_mass = soil_mass


class geo_inputs:
    def __init__(self, sf_width=30, side_length=316.228, depth=15, pez_cap=0, eff_area=0):
        self.sf_width = sf_width
        self.side_length = side_length
        self.depth = depth
        self.pez_area = sf_width * side_length
        self.pez_volume = sf_width * side_length * depth / 100
        self.pez_cap = pez_cap
        self.eff_area = eff_area


# summary output file
# opts 1 and 2 are for 1-in-15 year estimates
# opts 3 and 4 are for all 30 years of values
class OutputTable(object):
    def __init__(self, out_dir, out_file, opt):
        # Initialize path
        self.dir = out_dir
        self.base = out_file
        self.path = os.path.join(self.dir, self.base)

        # Initialize header
        if opt == 1:
            self.header = ["Line", "Batch Run ID", "EEC runoff only (lb/A)", "EEC runoff+drift 14-15 m (lb/A)"]
        elif opt == 2:
            self.header = ["Line", "Batch Run ID", "EEC (ug/L)", "EEC (lb/A)"]
        elif opt == 3:
            self.header = ["Line", "Batch Run ID", "Year", "EEC runoff only (lb/A)", "EEC runoff+drift 14-15 m (lb/A)"]
        else:
            self.header = ["Line", "Batch Run ID", "Year", "EEC (ug/L)", "EEC (lb/A)"]

        # Initialize file
        self.initialize_table()

    def initialize_table(self):
        # Create output directory if it does not exist
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        # Create file and write header
        with open(self.path, 'w') as f:
            f.write(",".join(self.header) + "\n")

    def update_table(self, line, run_id, yr, val1, val2):
        """ Insert run results into table """

        if yr == 99:
            out_data = [line, run_id, "{:.2E}".format(val1), "{:.2E}".format(val2)]
        else:
            out_data = [line, run_id, yr, "{:.2E}".format(val1), "{:.2E}".format(val2)]

        # Update file
        with open(self.path, 'a') as f:
            f.write(",".join(map(str, out_data)) + "\n")


# get inputs from stand-alone PWC run
def get_pwc_inputs(pwc_inputs):
    pwci = PWCInputs()
    hi = hydro_inputs()
    pwci.app_days.clear()
    pwci.app_months.clear()
    pwci.app_rates.clear()
    d_abs = "True"

    with open(pwc_inputs) as ts:
        str1 = ts.readline().rstrip('\n')
        str1 = ts.readline().rstrip('\n')
        pwci.chem_name = str1

        str1 = ts.readline().rstrip('\n')
        str2 = ts.readline().rstrip('\n')
        str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")

        if str2 == "True":
            pwci.Kd = float(tstr[0]) * 0.01
        else:
            pwci.Kd = float(tstr[0])

        for n in range(6, 14):
            str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")
        pwci.soil_t12 = float(tstr[0])
        if pwci.soil_t12 == 0:
            pwci.soil_t12 = 1e8

        str1 = ts.readline().rstrip('\n')
        str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")
        pwci.foliar_t12 = float(tstr[0])
        if pwci.foliar_t12 == 0:
            pwci.foliar_t12 = 1e8

        for n in range(16, 30):
            str1 = ts.readline().rstrip('\n')
        pwci.num_apps = int(str1)

        str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")
        for i in range(0, pwci.num_apps):
            if tstr[i] != "":
                pwci.app_days.append(int(tstr[i]))
            else:
                break

        str1 = ts.readline().rstrip('\n')
        if pwci.app_days:
            tstr = str1.split(",")
            for i in range(0, pwci.num_apps):
                pwci.app_months.append(int(tstr[i]))

        str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")
        for i in range(0, pwci.num_apps):
            pwci.app_rates.append(float(tstr[i]))

        for n in range(32, 43):
            str1 = ts.readline().rstrip('\n')

        tstr = str1.split(",")
        str1 = ts.readline().rstrip('\n')
        d_abs = str1

        for n in range(44, 49):
            str1 = ts.readline().rstrip('\n')
        pwci.scenario = str1

        for n in range(50, 77):
            str1 = ts.readline().rstrip('\n')

        i = 78
        if d_abs == "False":
            pwci.app_months = []
            pwci.app_days = []
            e_day = int(str1)
            str1 = ts.readline().rstrip('\n')
            e_month = int(str1)
            for i in range(0, pwci.num_apps):
                dt2 = dt.date(1960, e_month, e_day) + pd.DateOffset(days=int(tstr[i]))
                pwci.app_months.append(dt2.month)
                pwci.app_days.append(dt2.day)
            i = 79

        for n in range(i, 103):
            str1 = ts.readline().rstrip('\n')

        tstr = str1.split(",")
        hi.bulk_density = float(tstr[0])
        str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")
        hi.max_hold_cap = float(tstr[0])
        str1 = ts.readline().rstrip('\n')
        tstr = str1.split(",")
        hi.min_hold_cap = float(tstr[0])
        hi.avail_cap = hi.max_hold_cap - hi.min_hold_cap

    ts.close()
    return hi, pwci


# get inputs from PWC csv file that serves as an input file in a PWC batch run
def get_pwc_inputs2(pwc_inputs, input_dir, scen_dir, options):
    pwci = PWCInputs()
    hi = hydro_inputs()
    pwci.app_days.clear()
    pwci.app_months.clear()
    pwci.app_rates.clear()
    zts_file = ""
    aq_file = ""
    sa_file = ""
    str = ""

    # zts file is Run Name + _bin + ".zts"
    if "t" in options:
        zts_file = os.path.join(input_dir, pwc_inputs[0, 1] + "_10.zts")
    # aq file is Run Name + _7 + _Scenario Name + "_Custom_Parent_daily.csv"
    scen_name = pwc_inputs[0, 21]
    pwci.scenario = scen_name[:-4]

    if "a" in options:
        aq_file = os.path.join(input_dir, pwc_inputs[0, 1] + "_7_" + scen_name[:-4] + "_Custom_Parent_daily.csv")

    # need to find way to link semiaquatic runs (bin 11) to aq runs - look for same file name extension, using bin 10
    if "s" in options:
        str = "_10_" + scen_name[:-4] + "_Custom_Parent_daily.csv"
        files = os.listdir(input_dir)
        if files:
            lst = [i for i in files if i.endswith(str)]
            sa_file = os.path.join(input_dir, lst[0])
        else:
            print("Semiaquatic file does not exist")

    # pwci.Kd from SoprtionCoefficient(mL/g) and kocflag
    pwci.Kd = pwc_inputs[0, 2]
    if pwc_inputs[0, 3]:
        pwci.Kd *= 0.01

    # pwci.soil_t12 from SoilHalflife(days)
    pwci.soil_t12 = pwc_inputs[0, 11]
    if pwci.soil_t12 == 0:
        pwci.soil_t12 = 1e8

    # pwci.foliar_t12 from FoliarHalflife(day)
    pwci.foliar_t12 = pwc_inputs[0, 13]
    if pwc_inputs[0, 13] == 0:
        pwci.foliar_t12 = 1e8

    # pwci.num_apps from NumberofApplications
    pwci.num_apps = pwc_inputs[0, 74]

    # pwci.app_days, app_months, and app_rates from different columns
    for i in range(0, pwci.num_apps):
        pwci.app_days.append(pwc_inputs[0, i * 8 + 77])
        # if not pwc_inputs[0, i * 8 + 78]:
        #    pwci.app_months.append(0)
        # else:
        #
        pwci.app_months.append(pwc_inputs[0, i * 8 + 78])
        pwci.app_rates.append(pwc_inputs[0, i * 8 + 79])

    abs_rel = pwc_inputs[0, 75]

    # open scenario and get relevant info
    with open(os.path.join(scen_dir, scen_name)) as ts:
        for i in range(0, 28):
            str = ts.readline().rstrip('\n')

        if not abs_rel:
            tt = 54
            e_day = int(str)
            str = ts.readline().rstrip('\n')
            e_month = int(str)
            for i in range(0, pwci.num_apps):
                rday = pwci.app_days
                dt2 = dt.date(1960, e_month, e_day) + pd.DateOffset(days=rday[i])
                pwci.app_months[i] = dt2.month
                pwci.app_days[i] = dt2.day
        else:
            tt = 55

        for n in range(30, tt):
            str = ts.readline().rstrip('\n')

        tstr = str.split(",")
        hi.bulk_density = float(tstr[0])
        tstr = ts.readline().rstrip('\n').split(",")
        hi.max_hold_cap = float(tstr[0])
        tstr = ts.readline().rstrip('\n').split(",")
        hi.min_hold_cap = float(tstr[0])
        hi.avail_cap = hi.max_hold_cap - hi.min_hold_cap

    ts.close()

    # convert values in list from float to int
    pwci.app_days = [int(zz) for zz in pwci.app_days]
    pwci.app_months = [int(zz) for zz in pwci.app_months]

    return hi, pwci, zts_file, aq_file, sa_file


# develop spray drift values
def get_sdf(app_method, buffer, sdf_file):
    table = pd.read_csv(sdf_file)
    sdf = []
    for n in range(0, 30):
        sdf.append(n)
        sdf[n] = 0
        dist1 = round((n + buffer) * 3.2808, 4)
        dist2 = round((n + 1 + buffer) * 3.2808, 4)
        if dist2 < 997:
            bot = table[table["Dist_(ft)"] >= dist1].index[0]
            if bot == 0:
                bot = 1

            # get average deposition between dist1 and measurement just above dist1
            a = (table.loc[bot, app_method] - table.loc[bot - 1, app_method]) / (table.loc[bot, "Dist_(ft)"] -
                                                                                 table.loc[bot - 1, "Dist_(ft)"])
            b = table.loc[bot, app_method] - a * table.loc[bot, "Dist_(ft)"]
            dep = a * dist1 + b
            sdf[n] += (table.loc[bot, app_method] + dep) / 2 * (table.loc[bot, "Dist_(ft)"] - dist1)

            if table.loc[bot, "Dist_(ft)"] > dist2:  # span between spray drift distances is greater than 1 m
                a = (table.loc[bot, app_method] - table.loc[bot - 1, app_method]) / (table.loc[bot, "Dist_(ft)"] -
                                                                                     table.loc[bot - 1, "Dist_(ft)"])
                b = table.loc[bot, app_method] - a * table.loc[bot, "Dist_(ft)"]
                dep = a * dist2 + b
                sdf[n] += (table.loc[bot - 1, app_method] + dep) / 2 * (dist2 - table.loc[bot - 1, "Dist_(ft)"])
            else:
                # get average deposition between measurements up to the measurement just below dist2
                i = bot + 1
                while table.loc[i, "Dist_(ft)"] < dist2:
                    sdf[n] += ((table.loc[i, app_method] + table.loc[i - 1, app_method]) / 2 *
                               (table.loc[i, "Dist_(ft)"] - table.loc[i - 1, "Dist_(ft)"]))
                    i += 1

                # get average deposition between measurement just below dist2 and dist2
                a = (table.loc[i, app_method] - table.loc[i - 1, app_method]) / (table.loc[i, "Dist_(ft)"] -
                                                                                 table.loc[i - 1, "Dist_(ft)"])
                b = table.loc[i, app_method] - a * table.loc[i, "Dist_(ft)"]
                dep = a * dist2 + b
                sdf[n] += (table.loc[i - 1, app_method] + dep) / 2 * (dist2 - table.loc[i - 1, "Dist_(ft)"])

            # get average deposition
            sdf[n] /= (dist2 - dist1)
        else:
            sdf[n] = 0

    return sdf


# estimate the 1-in-10 year and 1-in-15 year values
def one_in_ten(vals, col_name):
    rslt1 = 0
    rslt2 = 0

    for n in range(0, 2):
        if n == 0:  # 1-in-10 yr estimate
            cnt = int(np.floor(0.1 * (len(vals) + 1)))
        else:  # 1-in-15 yr estimate
            cnt = int(np.floor(0.067 * (len(vals) + 1)))

        s = vals.nlargest(cnt + 1, col_name)
        # 1-in-10 yr estimate
        a1 = cnt / (len(vals) + 1)
        a2 = (cnt + 1) / (len(vals) + 1)
        slp = (s.values[cnt] - s.values[cnt - 1]) / (a2 - a1)
        intcpt = s.values[cnt] - slp * a2
        if n == 0:  # 1-in-10 yr estimate
            rslt1 = slp * 0.1 + intcpt
        else:  # 1-in-15 yr estimate
            rslt2 = slp * 0.067 + intcpt

    return rslt1, rslt2


# terrestrial plant calculations
def pat_terr(zts_file, run_name, hi, gi, pwci, sdf, make_figs, plant_edpt, esa, output_dir, line, tsum, tsum30):
    # get zts file
    col_names = ["year", "month", "day", "RUNF0", "ESLS0", "RFLX1", "EFLX1", "DCON1", "INFL0", "PRCP0"]
    table = pd.read_csv(zts_file, header=None, names=col_names, skiprows=3, delim_whitespace=True)
    if ~np.isnan(table.loc[0, "PRCP0"]):
        table.drop(["DCON1", "INFL0"], axis=1, inplace=True)

        # add columns to table and initialize the parameters
        # add parameter to indicate if an application occurred on the date or not, 1 = application occurred
        table["m_d"] = table["month"].astype(str) + "_" + table["day"].astype(str)
        table["app event"] = 0
        for n in range(0, pwci.num_apps):
            str1 = str(pwci.app_months[n]) + "_" + str(pwci.app_days[n])
            table.loc[table["m_d"] == str1, "app event"] = n + 1
        table.drop(["m_d"], axis=1, inplace=True)

        table["Runoff Depth cm"] = table["RUNF0"]
        table["Runoff Mass kg/ha"] = table["RFLX1"] * 100000
        table["Erosion Mass kg/ha"] = table["EFLX1"] * 100000
        table["Precipitation cm"] = table["PRCP0"]
        table["Runoff Volume m3/d"] = table["Runoff Depth cm"] / 100 * 10 * 10000
        table["Precip Volume m3/d"] = table["Precipitation cm"] / 100 * gi.pez_area
        table["Soil water content m3/m3"] = 0.0
        table["Runoff+Infil m3/d"] = 0.0
        table["mass runon kg"] = table["Runoff Mass kg/ha"] * 10
        table["mass sediment kg"] = table["Erosion Mass kg/ha"] * 10

        for n in range(0, 30):
            nstr = "foliar spray kg " + str(n) + "_" + str(n + 1)
            table[nstr] = 0

        table["foliar washoff kg"] = 0
        table["K1 d-1"] = 0
        table["K2 kg/m3/d"] = 0
        table["mass tpez water kg"] = 0
        table["mass tpez soil kg"] = 0

        for n in range(0, 30):
            nstr = "foliar mass remain kg " + str(n) + "_" + str(n + 1)
            table[nstr] = 0
        table["runoff eec kg/ha"] = 0

        for n in range(0, 30):
            nstr = "spray+runoff eec kg/ha " + str(n) + "_" + str(n + 1)
            table[nstr] = 0

        column_names = table.columns
        tab2 = table.values
        # insert daily calculations
        for n in range(0, len(table)):
            print("Terrestrial model, processing day {}".format(n))
            if tab2[n, 13] + tab2[n, 14] == 0:
                tab2[n, 15] = hi.min_hold_cap
                tab2[n, 16] = 0.0
            else:
                if n == 0:
                    prev_day = hi.max_hold_cap
                else:
                    prev_day = tab2[n - 1, 15]
                prev_day += (tab2[n, 13] + tab2[n, 14]) / gi.pez_cap
                if prev_day - hi.max_hold_cap > 0:
                    tab2[n, 15] = hi.max_hold_cap
                else:
                    tab2[n, 15] = prev_day
                tab2[n, 16] = (tab2[n, 13] + tab2[n, 14] - (tab2[n, 15] - tab2[n - 1, 15]) * gi.pez_cap)
                if abs(tab2[n, 16]) < 1e-14:
                    tab2[n, 16] = 0

            # foliar mass due to spray
            foliar_sum = 0
            for i in range(0, 30):
                if tab2[n, 8] == 0:
                    if n == 0:
                        tab2[n, i + 19] = 0
                    else:
                        tab2[n, i + 19] = tab2[n - 1, i + 54] * np.exp(-1 * np.log(2) / pwci.foliar_t12)
                else:
                    if n == 0:
                        tab2[n, i + 19] = 0
                    else:
                        ar = pwci.app_rates[int(tab2[n, 8]) - 1]
                        tab2[n, i + 19] = (tab2[n - 1, i + 54] * np.exp(-1 * np.log(2) / pwci.foliar_t12) + ar *
                                           sdf[i] * gi.side_length * 1 / 10000)
                foliar_sum += tab2[n, i + 19]

            if tab2[n, 12] != 0:
                tab2[n, 49] = (foliar_sum * (1 - np.exp(-1 * tab2[n, 12] * pwci.washoff)))

            # calculate K1 and K2. denom is used in both - I think this is calculated wrong in Excel tool
            # denom = (tab2.loc[n, "Soil water content m3/m3"] / gi.depth + pwci.Kd * hi.bulk_density) * gi.pez_volume
            denom = (tab2[n, 15] + pwci.Kd * hi.bulk_density) * gi.pez_volume
            tab2[n, 50] = tab2[n, 16] / denom + np.log(2) / pwci.soil_t12

            tab2[n, 51] = tab2[n, 17] + tab2[n, 18]
            if n != 0:
                tab2[n, 51] += (tab2[n - 1, 49] + tab2[n - 1, 52] + tab2[n - 1, 53])
                tab2[n, 51] /= denom

            # calculate mass in water and soil - I think these are calculated wrong in Excel
            # tab2.loc[n, "mass tpez water kg"] = (tab2.loc[n, "K2 kg/m3/d"] * np.exp(-1 * tab2.loc[n,"K1 d-1"]) *
            #                                      tab2.loc[n, "Soil water content m3/m3"] / gi.depth * gi.pez_volume)
            # tab2.loc[n, "mass tpez soil kg"] = (tab2.loc[n, "mass tpez water kg"] * pwci.Kd / 1000 * hi.soil_mass /
            #                                     (tab2.loc[n, "Soil water content m3/m3"] / gi.depth * gi.pez_volume))

            tab2[n, 52] = (tab2[n, 51] * np.exp(-1 * tab2[n, 50]) * tab2[n, 15] * gi.pez_volume)

            tab2[n, 53] = (tab2[n, 52] * pwci.Kd / 1000 * hi.soil_mass / (tab2[n, 15] * gi.pez_volume))

            # calculate runoff EEC
            tab2[n, 84] = ((tab2[n, 52] + tab2[n, 53]) / (gi.pez_area / 10000))

            # calculate remaining foliar mass and EECs
            for i in range(0, 30):
                if tab2[n, 12] == 0:
                    tab2[n, i + 54] = tab2[n, i + 19]
                else:
                    tab2[n, i + 54] = tab2[n, i + 19] * np.exp(-1 * tab2[n, 12] * pwci.washoff)
                tab2[n, i + 85] = tab2[n, 84] + tab2[n, i + 54] / (gi.side_length * 1 / 10000)

        ter_table = pd.DataFrame(data=tab2, columns=column_names)
        # create summary file
        numyrs = len(ter_table['year'].unique())
        col_names = np.empty(shape=[32], dtype='object')
        col_names[0] = 'year'
        col_names[1] = "peak runoff lb/A"
        for n in range(0, 30):
            col_names[n + 2] = "peak eec " + str(n) + "_" + str(n + 1) + " m lb/A"

        sum_tab2 = np.empty(shape=[numyrs, 32])
        n = 0
        for yr in ter_table["year"].unique():
            table_sum = ter_table[ter_table["year"] == yr]
            print("Processing year {}...".format(int(yr)))
            # Analyze the data
            sum_tab2[n, 0] = int(yr)
            sum_tab2[n, 1] = np.max(table_sum["runoff eec kg/ha"]) / 1.12
            for i in range(0, 30):
                nstr2 = "spray+runoff eec kg/ha " + str(i) + "_" + str(i + 1)
                sum_tab2[n, i + 2] = np.max(table_sum[nstr2]) / 1.12
            n += 1

        sum_stat = np.empty(shape=[2, 32])
        sum_stat[0, 0] = 99
        sum_stat[1, 0] = 99
        sum_stat[0, 1], sum_stat[1, 1] = one_in_ten(pd.DataFrame(sum_tab2[:, 1], columns=["runoff eec kg/ha"]),
                                                    "runoff eec kg/ha")
        for i in range(0, 30):
            nstr2 = "spray+runoff eec kg/ha " + str(i) + "_" + str(i + 1)
            sum_stat[0, i + 2], sum_stat[1, i + 2] = one_in_ten(pd.DataFrame(sum_tab2[:, i + 2], columns=[nstr2]),
                                                                nstr2)

        sum_tab2 = np.append(sum_tab2, sum_stat, axis=0)

        # create terrestrial EEC file and EEC summary file with results
        print("Saving terrestrial data for run {} ...".format(run_name))
        ter_table.to_csv(os.path.join(output_dir, (run_name + "_terr_results.csv")), index=False)

        # create an index to use with summary file
        indx = np.empty(numyrs + 2, dtype='object')
        for i in range(0, numyrs):
            indx[i] = str(sum_tab2[i, 0])
        indx[numyrs] = "1-in-10 yr"
        indx[numyrs + 1] = "1-in-15 yr"

        sum_data = pd.DataFrame(data=sum_tab2, columns=col_names, index=indx)
        sum_data.drop(["year"], axis=1, inplace=True)
        sum_data.index.name = "Year"
        print("Saving terrestrial summary data for run {} ...".format(run_name))
        sum_data.to_csv(os.path.join(output_dir, (run_name + "_terr_summary.csv")))

        # save data to summary files - currently saves edge of field+spray drift estimate
        if line != -1:  # if -1, then not a batch file run
            if esa:
                tsum.update_table(line, run_name, 99, sum_tab2[numyrs + 1, 1], sum_tab2[31, 16])
            else:
                tsum.update_table(line, run_name, 99, sum_tab2[numyrs, 1], sum_tab2[30, 16])
            for n in range(0, numyrs):
                tsum30.update_table((line - 1) * numyrs + n, run_name, n + 1, sum_tab2[n, 1], sum_tab2[n, 16])

        # create terrestrial RQ tables
        rq_colnames = ["SE/VV", "Species", "Runoff EEC (lb/A)", "Runoff Non-listed RQ", "Runoff Listed RQ",
                       "Outer Edge EEC (lb/A)", "Outer Edge Non-listed RQ", "Outer Edge Listed RQ"]
        # determine unique crops
        rq_tab = np.empty(shape=[len(plant_edpt), 8], dtype='object')
        for i in range(0, len(plant_edpt)):
            rq_tab[i, 0] = plant_edpt.iloc[i, 0]
            rq_tab[i, 1] = plant_edpt.iloc[i, 1]
            if esa:
                rq_tab[i, 2] = sum_tab2[numyrs + 1, 1]  # these are the 1-in-15 yr values for ESA
                rq_tab[i, 5] = sum_tab2[numyrs + 1, 31]
            else:
                rq_tab[i, 2] = sum_tab2[numyrs, 1]  # these are the 1-in-10 yr values, for QC against Excel version
                rq_tab[i, 5] = sum_tab2[numyrs, 31]
            for z in range(0, 2):
                if ~np.isnan(plant_edpt.iloc[i, (z * 5 + 5)]):
                    for t in range(0, 2):
                        rq = np.round(rq_tab[i, (t * 3 + 2)] / plant_edpt.iloc[i, (z * 5 + 5)], 3)
                        if rq < 0.01:
                            rq_tab[i, (t * 3 + 3 + z)] = "< 0.01"
                        elif plant_edpt.iloc[i, (z * 5 + 4)] == "<":
                            rq_tab[i, (t * 3 + 3 + z)] = "> " + str(np.round(rq, 2))
                        elif plant_edpt.iloc[i, (z * 5 + 4)] == ">":
                            rq_tab[i, (t * 3 + 3 + z)] = "< " + str(np.round(rq, 2))
                        else:
                            rq_tab[i, (t * 3 + 3 + z)] = str(np.round(rq, 2))
                else:
                    rq_tab[i, (3 + z)] = "NA"
                    rq_tab[i, (6 + z)] = "NA"

        print("Saving terrestrial RQ summary for run {} ...".format(run_name))
        pd.DataFrame(data=rq_tab, columns=rq_colnames).to_csv(os.path.join(output_dir, (run_name + "_terr_rqs.csv")),
                                                              index=False)

        # create figures
    #        if make_figs:
    # dts = np.empty(shape=[len(ter_table)], dtype='object')
    # for n in range(0,len(ter_table)):
    #    dts[n] = str(1900+int(tab2[n,0])) + "-" + str(int(tab2[n, 1])) + "-" + str(int(tab2[n,2]))
    # converted_dates = pltdt.datestr2num(dts)
    # plt.figure(num=1,figsize=(10.0,7.0))

    # plt.subplot(211)
    # plt.plot_date(converted_dates, tab2[:, 85], "b-")       # near edge of TPEZ
    # plt.plot_date(converted_dates, tab2[:, 114], "r-")      # far edge of TPEZ
    # plt.title("Concentrations Over Time in TPEZ, Drift + Runoff", fontsize=8)
    # plt.xlabel("Year", fontsize='x-small')
    # plt.ylabel("Concentration (lb/A)", fontsize='x-small')
    # plt.xticks(fontsize='x-small')
    # plt.yticks(fontsize='x-small')
    # plt.legend(("Near-edge", "Far-edge"), loc='best', fontsize='xx-small', ncol = 2)

    # plt.subplot(212)
    # plt.plot_date(converted_dates, tab2[:, 84], "b-")       # runoff across TPEZ
    # plt.title("Concentrations Over Time in TPEZ, Runoff only", fontsize=8)
    # plt.xlabel("Year", fontsize='x-small')
    # plt.ylabel("Concentration (lb/A)", fontsize='x-small')
    # plt.xticks(fontsize='x-small')
    # plt.yticks(fontsize='x-small')
    # plt.legend("Across_TPEZ", loc='best', fontsize='xx-small', ncol = 1)

    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir,(run_name + "terr_figs.pdf")), format="PDF")
    else:
        print("Precipitation data not contained in ZTS file, cannot calculate TPEZ EECs")


def pat_semiaq(semi_csv, run_name, pwci, sai, make_figs, plant_edpt, esa, output_dir, line, ssum, ssum30):
    col_names = ["depth m", "avg aq water kg/m3", "avg aq benthic kg/m3", "peak aq water kg/m3"]
    satable = pd.read_csv(semi_csv, header=None, index_col=False, names=col_names, skiprows=5,
                          delim_whitespace=False)
    # add date and year columns
    satable["Date"] = ""
    satable["Year"] = 0
    satable["avg aq water ug/L"] = 0
    satable["EEC lb/A"] = 0
    indx_order = [4, 5, 0, 1, 2, 3, 6, 7]
    satable = satable[[satable.columns[i] for i in indx_order]]

    col_names = satable.columns
    tab2 = satable.values

    dy = pd.datetime.strptime("12/31/1960", '%m/%d/%Y').date()
    for n in range(0, len(satable)):
        print("Semi-aquatic model, processing day {}".format(n))
        # insert date and year
        dat = dy + pd.DateOffset(days=n + 1)
        tab2[n, 0] = str(dat.month) + "/" + str(dat.day) + "/" + str(dat.year)
        tab2[n, 1] = int(dat.year)
        if tab2[n, 2] >= 0.005:
            tab2[n, 6] = tab2[n, 3] * 1000 * 1e6 / 1000
            tab2[n, 7] = tab2[n, 3] * 10000 * tab2[n, 2]
        tab2[n, 7] += tab2[n, 4] * ((pwci.Kd / 1000) * sai.bulk_density * sai.volume_sed + sai.volume_water)
        tab2[n, 7] = tab2[n, 7] / 10000 * 10000 / 1.12

    satable2 = pd.DataFrame(data=tab2, columns=col_names)

    # create summary file
    numyrs = len(satable2['Year'].unique())
    col_names2 = np.empty(shape=[3], dtype='object')
    col_names2[0] = 'Year'
    col_names2[1] = "EEC (ug/L)"
    col_names2[2] = "EEC (lb/A)"

    sum_tab2 = np.empty(shape=[numyrs, 3])
    n = 0
    for yr in satable2["Year"].unique():
        table_sum = satable2[satable2["Year"] == yr]
        print("Processing year {} ...".format(yr))
        # Analyze the data
        sum_tab2[n, 0] = yr
        sum_tab2[n, 1] = np.max(table_sum["avg aq water ug/L"])
        sum_tab2[n, 2] = np.max(table_sum["EEC lb/A"])
        n += 1

    sum_stat = np.empty(shape=[2, 3])
    sum_stat[0, 0] = 99
    sum_stat[1, 0] = 99
    sum_stat[0, 1], sum_stat[1, 1] = one_in_ten(pd.DataFrame(sum_tab2[:, 1], columns=["avg aq water ug/L"]),
                                                "avg aq water ug/L")
    sum_stat[0, 2], sum_stat[1, 2] = one_in_ten(pd.DataFrame(sum_tab2[:, 2], columns=["EEC lb/A"]), "EEC lb/A")

    sum_tab2 = np.append(sum_tab2, sum_stat, axis=0)

    # create semi aquatic file and summary file with results
    print("Saving wetlands data for run {} ...".format(run_name))
    satable2.to_csv(os.path.join(output_dir, (run_name + "_semiaq_results.csv")), index=False)

    # create an index to use with summary file
    indx = np.empty(shape=[numyrs + 2], dtype='object')
    for i in range(0, 30):
        indx[i] = str(sum_tab2[i, 0])
    indx[numyrs] = "1-in-10 yr"
    indx[numyrs + 1] = "1-in-15 yr"

    sum_data = pd.DataFrame(data=sum_tab2, columns=col_names2, index=indx)
    sum_data.drop(["Year"], axis=1, inplace=True)
    sum_data.index.name = "Year"
    print("Saving wetlands summary data for run {} ...".format(run_name))
    sum_data.to_csv(os.path.join(output_dir, (run_name + "_semiaq_summary.csv")))

    # save data to summary files
    if line != -1:  # -1, not running a batch file
        if esa:
            ssum.update_table(line, run_name, 99, sum_tab2[numyrs + 1, 1], sum_tab2[31, 2])
        else:
            ssum.update_table(line, run_name, 99, sum_tab2[numyrs, 1], sum_tab2[30, 2])
        for n in range(0, 30):
            ssum30.update_table((line - 1) * 30 + n, run_name, n + 1, sum_tab2[n, 1], sum_tab2[n, 2])

    # create semiaquatic RQ tables
    rq_colnames = ["SE/VV/AQ", "Species", "Wetlands EEC", "Non-listed Wetlands RQ", "Listed Wetlands RQ"]
    # determine unique crops
    rq_tab = np.empty(shape=[len(plant_edpt), 5], dtype='object')
    for i in range(0, len(plant_edpt)):
        rq_tab[i, 0] = plant_edpt.iloc[i, 0]
        rq_tab[i, 1] = plant_edpt.iloc[i, 1]
        if esa:
            t = numyrs + 1
        else:
            t = numyrs
        if rq_tab[i, 0] == "AQ":
            rq_tab[i, 2] = sum_tab2[t, 1]
        else:
            rq_tab[i, 2] = sum_tab2[t, 2]
        for z in range(0, 2):
            if ~np.isnan(plant_edpt.iloc[i, (z * 5 + 5)]):
                rq = np.round(rq_tab[i, 2] / plant_edpt.iloc[i, (z * 5 + 5)], 3)
                if rq < 0.01:
                    rq_tab[i, (3 + z)] = "< 0.01"
                elif plant_edpt.iloc[i, (z * 5 + 4)] == "<":
                    rq_tab[i, (3 + z)] = "> " + str(np.round(rq, 2))
                elif plant_edpt.iloc[i, (z * 5 + 4)] == ">":
                    rq_tab[i, (3 + z)] = "< " + str(np.round(rq, 2))
                else:
                    rq_tab[i, (3 + z)] = str(np.round(rq, 2))
            else:
                rq_tab[i, (3 + z)] = "NA"

    print("Saving wetlands RQ summary for run {} ...".format(run_name))
    pd.DataFrame(data=rq_tab, columns=rq_colnames).to_csv(os.path.join(output_dir, (run_name + "_semiaq_rqs.csv")),
                                                          index=False)

    # create figures
    # if make_figs:
    #    dts = np.empty(shape=[len(satable)], dtype='object')
    #    for n in range(0, len(satable)):
    #        dts[n] = str(1900 + int(tab2[n, 0])) + "-" + str(int(tab2[n, 1])) + "-" + str(int(tab2[n, 2]))
    #    converted_dates = pltdt.datestr2num(dts)
    #    plt.figure(num=1, figsize=(10.0, 7.0))


def pat_aq(aq_csv, run_name, make_figs, plant_edpt, esa, output_dir):
    col_names = ["depth m", "avg aq water kg/m3", "avg aq benthic kg/m3", "peak aq water kg/m3"]
    aqtable = pd.read_csv(aq_csv, header=None, index_col=False, names=col_names, skiprows=5,
                          delim_whitespace=False)
    # add date and year columns
    aqtable["Date"] = ""
    aqtable["Year"] = 0
    aqtable["avg aq water ug/L"] = 0
    indx_order = [4, 5, 0, 1, 2, 3, 6]
    aqtable = aqtable[[aqtable.columns[i] for i in indx_order]]

    col_names = aqtable.columns
    tab2 = aqtable.values

    dy = pd.datetime.strptime("12/31/1960", '%m/%d/%Y').date()
    for n in range(0, len(aqtable)):
        print("Aquatic model, processing day {}".format(n))
        # insert date and year
        dat = dy + pd.DateOffset(days=n + 1)
        tab2[n, 0] = str(dat.month) + "/" + str(dat.day) + "/" + str(dat.year)
        tab2[n, 1] = int(dat.year)
        tab2[n, 6] = tab2[n, 3] * 1000 * 1e6 / 1000

    aqtable2 = pd.DataFrame(data=tab2, columns=col_names)

    # create summary file
    numyrs = len(aqtable2['Year'].unique())
    col_names2 = np.empty(shape=[2], dtype='object')
    col_names2[0] = 'Year'
    col_names2[1] = "EEC (ug/L)"

    sum_tab2 = np.empty(shape=[numyrs, 2])
    n = 0
    for yr in aqtable2["Year"].unique():
        table_sum = aqtable2[aqtable2["Year"] == yr]
        print("Processing year {} ...".format(yr))
        # Analyze the data
        sum_tab2[n, 0] = yr
        sum_tab2[n, 1] = np.max(table_sum["avg aq water ug/L"])
        n += 1

    sum_stat = np.empty(shape=[2, 2])
    sum_stat[0, 0] = 99
    sum_stat[1, 0] = 99
    sum_stat[0, 1], sum_stat[1, 1] = one_in_ten(pd.DataFrame(sum_tab2[:, 1], columns=["avg aq water ug/L"]),
                                                "avg aq water ug/L")

    sum_tab2 = np.append(sum_tab2, sum_stat, axis=0)

    # create aquatic file and summary file with results
    print("Saving aquatic data for run {} ...".format(run_name))
    aqtable2.to_csv(os.path.join(output_dir, (run_name + "_aq_results.csv")), index=False)

    # create an index to use with summary file
    indx = np.empty(shape=[numyrs + 2], dtype='object')
    for i in range(0, numyrs):
        indx[i] = str(sum_tab2[i, 0])
    indx[numyrs] = "1-in-10 yr"
    indx[numyrs + 1] = "1-in-15 yr"

    sum_data = pd.DataFrame(data=sum_tab2, columns=col_names2, index=indx)
    sum_data.drop(["Year"], axis=1, inplace=True)
    sum_data.index.name = "Year"
    print("Saving aquatic summary data for run {} ...".format(run_name))
    sum_data.to_csv(os.path.join(output_dir, (run_name + "_aq_summary.csv")))

    # create aquatic RQ tables
    rq_colnames = ["AQ", "Species", "Aquatic EEC", "Non-listed Aquatic RQ", "Listed Aquatic RQ"]
    # determine unique crops
    rq_tab = np.empty(shape=[len(plant_edpt), 5], dtype='object')
    for i in range(0, len(plant_edpt)):
        rq_tab[i, 0] = plant_edpt.iloc[i, 0]
        rq_tab[i, 1] = plant_edpt.iloc[i, 1]
        if esa:
            t = numyrs + 1
        else:
            t = numyrs
        rq_tab[i, 2] = sum_tab2[t, 1]
        for z in range(0, 2):
            if ~np.isnan(plant_edpt.iloc[i, (z * 5 + 5)]):
                rq = np.round(rq_tab[i, 2] / plant_edpt.iloc[i, (z * 5 + 5)], 3)
                if rq < 0.01:
                    rq_tab[i, (3 + z)] = "< 0.01"
                elif plant_edpt.iloc[i, (z * 5 + 4)] == "<":
                    rq_tab[i, (3 + z)] = "> " + str(np.round(rq, 2))
                elif plant_edpt.iloc[i, (z * 5 + 4)] == ">":
                    rq_tab[i, (3 + z)] = "< " + str(np.round(rq, 2))
                else:
                    rq_tab[i, (3 + z)] = str(np.round(rq, 2))
            else:
                rq_tab[i, (3 + z)] = "NA"

    print("Saving aquatic RQ summary for run {} ...".format(run_name))
    pd.DataFrame(data=rq_tab, columns=rq_colnames).to_csv(os.path.join(output_dir, (run_name + "_aq_rqs.csv")),
                                                          index=False)

    # create figures


#    if make_figs:
#        dts = np.empty(shape=[len(aqtable)], dtype='object')
#        for n in range(0, len(aqtable)):
#            dts[n] = str(1900 + int(tab2[n, 0])) + "-" + str(int(tab2[n, 1])) + "-" + str(int(tab2[n, 2]))
#        converted_dates = pltdt.datestr2num(dts)
#        plt.figure(num=1, figsize=(10.0, 7.0))


def main():
    # Data paths
    # TODO - suggestions
    # Status updates too frequent
    tic = dt.datetime.now()
    print("Start time: %s" % (tic))

    run_dir = os.path.join("..", "qaqc_test")
    input_dir = os.path.join(run_dir, "test1")  # directory where PAT input files are located
    output_dir = os.path.join(input_dir, "Output")  # directory for all output files
    make_figs = False  # TRUE if you want to create figures
    batch_run = False  # TRUE if you are doing a batch run
    batch_file = os.path.join(run_dir, "PAT Batch Run test2.csv")  # if doing batch run, name of input file
    use_pwc_batch = False  # use this flag to let program know that you will be using the
    # PWC batch file for inputs
    scen_dir = "test2"  # if using PWC batch input file for info, location of ESA scenarios
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sdf_file = os.path.join(run_dir, "sdf.csv")  # file for spray drift info, normally you don't have to change
    endpoints = os.path.join(run_dir, "plant endpoints.csv")  # file name with plant endponts
    tsum = ""  # PAT terrestrial summary file, used for batch runs
    tsum30 = ""  # PAT terrestrial 30 year summary file, used for batch runs
    ssum = ""  # PAT semiaquatic summary file, used for batch runs
    ssum30 = ""  # PAT terrestrial 30 year summary file, used for batch runs

    # if doing a single run, file names in following variables
    run_name = ""  # name of PAT output file
    zts_file = os.path.join(input_dir, "test1a.zts")
    pwc_input = os.path.join(input_dir, "test1a.SWI")
    semi_csv = os.path.join(input_dir, "test1a_CArightofwayRLF_V2_Custom_Parent_daily.csv")
    aq_csv = os.path.join(input_dir, "test1a_CArightofwayRLF_V2_Pond_Parent_daily.csv")
    app_method = "Aerial_Fine_to_Medium"

    options = "tsa"  # PAT options = t = terrestrial, a = aquatics, s = semiaquatic,
    # tsa = all, order of the letters doesn't matter
    esa = False  # flag if doing eas or fifra run - fifra - 1 in 10, esa 1 in 15
    buffer_setback = 0  # buffer distance, if applicable

    # next set of classes are designed to store inputs for PAT run
    gi = geo_inputs()
    hi = hydro_inputs()
    pwci = PWCInputs()
    sai = SemiPWCInputs()

    """ All path and parameter variables are set before this line. The rest is program functionality """

    # get plant endpoints
    plant_edpt = pd.read_csv(endpoints, index_col=False)

    # check if batch run; if so, grab info for runs
    if batch_run:
        batch_inputs = pd.read_csv(batch_file, index_col=False)
        if "t" in options:
            tsum = OutputTable(output_dir, "PAT Terrestrial Summary.csv", 1)
            tsum30 = OutputTable(output_dir, "PAT Terrestrial Summary 30.csv", 3)
        if "s" in options:
            ssum = OutputTable(output_dir, "PAT Semiaquatic Summary.csv", 2)
            ssum30 = OutputTable(output_dir, "PAT Semiaquatic Summary 30.csv", 4)

        for n in range(0, len(batch_inputs)):
            run_name = batch_inputs.loc[n, "Run ID"]
            if use_pwc_batch:
                if n == 0:
                    pwc_input = os.path.join(input_dir, batch_inputs.loc[n, "PWC Input File"])
                    pwc_vals = pd.read_csv(pwc_input, index_col=False)
                hi, pwci, zts_file, aq_csv, semi_csv = get_pwc_inputs2(
                    pwc_vals[pwc_vals["Run Name"] == run_name].values,
                    input_dir, scen_dir, options)
            else:
                pwc_input = os.path.join(input_dir, batch_inputs.loc[n, "PWC Input File"])
                hi, pwci = get_pwc_inputs(pwc_input)

            hi.soil_mass = gi.pez_volume * hi.bulk_density * 1000
            gi.pez_cap = hi.avail_cap * gi.pez_volume
            if buffer_setback > 30:
                gi.eff_area = 0
            else:
                gi.eff_area = (gi.sf_width - buffer_setback) * gi.side_length
            sai.bulk_density = hi.bulk_density * 1000
            # open spray drift deposition file and develop deposition values for first 30 m
            pwci.app_method = batch_inputs.loc[n, "Spray DSD"]
            sdf = get_sdf(pwci.app_method, buffer_setback, sdf_file)

            if "t" in options:
                # get zts file and generate terrestrial outputs
                if not use_pwc_batch:
                    zts_file = os.path.join(input_dir, batch_inputs.loc[n, "PWC ZTS File Name"])
                print("Conducting terrestrial analysis for {}".format(zts_file))
                edpt = plant_edpt[(plant_edpt["Type"] == "SE") | (plant_edpt["Type"] == "VV")]
                pat_terr(zts_file, run_name, hi, gi, pwci, sdf, make_figs, edpt, esa, output_dir, n + 1, tsum, tsum30)

            if "s" in options:
                # get PWC semiaquatic csv file
                if not use_pwc_batch:
                    semi_csv = os.path.join(input_dir, batch_inputs.loc[n, "PWC Semi-Aquatic File"])
                print("Conducting semi-aquatic analysis for {}".format(semi_csv))
                pat_semiaq(semi_csv, run_name, pwci, sai, make_figs, plant_edpt, esa, output_dir, n + 1, ssum, ssum30)

            if "a" in options:
                # get PWC aquatic csv file
                if not use_pwc_batch:
                    aq_csv = os.path.join(input_dir, batch_inputs.loc[n, "PWC Aquatic File"])
                print("Conducting aquatic analysis for {}".format(aq_csv))
                edpt = plant_edpt[plant_edpt["Type"] == "AQ"]
                pat_aq(aq_csv, run_name, make_figs, edpt, esa, output_dir)
    else:
        # get pwc inputs
        # run_name = ""
        hi, pwci = get_pwc_inputs(pwc_input)
        hi.soil_mass = gi.pez_volume * hi.bulk_density * 1000
        gi.pez_cap = hi.avail_cap * gi.pez_volume
        if buffer_setback > 30:
            gi.eff_area = 0
        else:
            gi.eff_area = (gi.sf_width - buffer_setback) * gi.side_length
        sai.bulk_density = hi.bulk_density * 1000

        # open spray drift deposition file and develop deposition values for first 30 m
        if app_method != "":
            pwci.app_method = app_method
        sdf = get_sdf(pwci.app_method, buffer_setback, sdf_file)

        if "t" in options:
            # get zts file and generate terrestrial outputs
            print("Conducting terrestrial analysis for {}".format(zts_file))
            edpt = plant_edpt[(plant_edpt["Type"] == "SE") | (plant_edpt["Type"] == "VV")]
            pat_terr(zts_file, run_name, hi, gi, pwci, sdf, make_figs, edpt, esa, output_dir, -1, tsum, tsum30)

        if "s" in options:
            # get PWC semiaquatic csv file
            print("Conducting semi-aquatic analysis for {}".format(semi_csv))
            pat_semiaq(semi_csv, run_name, pwci, sai, make_figs, plant_edpt, esa, output_dir, -1, ssum, ssum30)

        if "a" in options:
            # get PWC aquatic csv file
            print("Conducting aquatic analysis for {}".format(aq_csv))
            edpt = plant_edpt[plant_edpt["Type"] == "AQ"]
            pat_aq(aq_csv, run_name, make_figs, edpt, esa, output_dir)

    toc = dt.datetime.now()
    print("End time: %s" % (toc))
    print("Execution time: %s" % (toc - tic))


main()
