#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timedelta
from streamlit_calendar import calendar  # Make sure to install streamlit-calendar

# Set page configuration
st.set_page_config(page_title="Nurse Scheduling Problem Solver", layout="wide")

st.title("NSP Solver by HH")

# File uploaders
cover_requirements_file = st.file_uploader("Upload Cover Requirements CSV", type="csv", key="cover_requirements")
days_off_file = st.file_uploader("Upload Days Off CSV", type="csv", key="days_off")
shift_off_requests_file = st.file_uploader("Upload Shift Off Requests CSV", type="csv", key="shift_off_requests")
shift_on_requests_file = st.file_uploader("Upload Shift On Requests CSV", type="csv", key="shift_on_requests")
staff_file = st.file_uploader("Upload Staff CSV", type="csv", key="staff")

# Proceed only if all files are uploaded
if (cover_requirements_file and days_off_file and shift_off_requests_file 
    and shift_on_requests_file and staff_file):

    st.success("All files uploaded successfully. Processing data...")

    # Read the uploaded files into DataFrames
    cover_requirements = pd.read_csv(cover_requirements_file)
    shift_off_requests = pd.read_csv(shift_off_requests_file)
    shift_on_requests = pd.read_csv(shift_on_requests_file)
    staff = pd.read_csv(staff_file)

    # Load cover requirements as a dictionary
    cover_requirements_dict = {
        (day, shift.strip()): requirement for day, shift, requirement in cover_requirements.itertuples(index=False)
    }

    # Extract necessary details from the data for optimization
    nurses = staff['# ID'].astype(str).tolist()
    days = cover_requirements['# Day'].unique().tolist()
    shift_types = cover_requirements[' ShiftID'].unique().tolist()
    shift_types = [shift.strip() for shift in shift_types]  # Remove any leading/trailing spaces

    # Read the days_off.csv file line by line to handle variable-length rows
    days_off_dict = {}
    days_off_file.seek(0)  # Reset file pointer to the beginning
    lines = days_off_file.readlines()
    lines = [line.decode('utf-8').strip() for line in lines]

    for line in lines[1:]:  # Skip header if present
        parts = line.strip().split(',')
        employee_id = parts[0].strip()
        days_off = list(map(int, parts[1:])) if len(parts) > 1 else []
        days_off_dict[employee_id] = days_off

    # Parse max shifts and other constraints from the staff data
    max_shifts_dict = {}
    for _, row in staff.iterrows():
        nurse_id = str(row['# ID'])
        max_shifts_info = str(row[' MaxShifts']).split('|')
        max_shifts = {shift.split('=')[0].strip(): int(shift.split('=')[1]) for shift in max_shifts_info}
        max_shifts_dict[nurse_id] = max_shifts

    # Extract other staff information
    max_total_minutes = staff.set_index('# ID')[' MaxTotalMinutes'].to_dict()
    max_total_minutes = {str(k): v for k, v in max_total_minutes.items()}
    max_consecutive_shifts = staff.set_index('# ID')[' MaxConsecutiveShifts'].to_dict()
    max_consecutive_shifts = {str(k): v for k, v in max_consecutive_shifts.items()}

    # Decision Variables: x[nurse_id][day_id][shift_id] = 1 if nurse is assigned to shift on a specific day
    x = {
        (nurse_id, day_id, shift_id): LpVariable(name=f"x_{nurse_id}_{day_id}_{shift_id}", cat='Binary')
        for nurse_id in nurses
        for day_id in days
        for shift_id in shift_types
    }

    # Create the optimization problem
    model = LpProblem(name="nurse-scheduling", sense=LpMinimize)

    # Objective Function: Minimize unmet requests (can be modified to other priorities)
    shift_on_requests['# EmployeeID'] = shift_on_requests['# EmployeeID'].astype(str)
    shift_on_requests_dict = shift_on_requests.set_index(['# EmployeeID', ' Day', ' ShiftID'])[' Weight'].to_dict()
    shift_off_requests['# EmployeeID'] = shift_off_requests['# EmployeeID'].astype(str)
    shift_off_requests_dict = shift_off_requests.set_index(['# EmployeeID', ' Day', ' ShiftID'])[' Weight'].to_dict()

    # Step 1: Add auxiliary variables for excess nurses assigned beyond the cover requirements
    excess_vars = LpVariable.dicts("excess", [(day_id, shift_id) for day_id in days for shift_id in shift_types], lowBound=0, cat='Continuous')

    # Step 2: Add constraints to define excess variables
    for day_id in days:
        for shift_id in shift_types:
            if (day_id, shift_id) in cover_requirements_dict:
                requirement = cover_requirements_dict[(day_id, shift_id)]
                # Use correct indices
                model += (lpSum(x[(nurse_id, day_id, shift_id)] for nurse_id in nurses) - requirement <= excess_vars[(day_id, shift_id)], f"excess_cover_{day_id}_{shift_id}")
                model += (excess_vars[(day_id, shift_id)] >= 0, f"excess_non_negative_{day_id}_{shift_id}")

    # Step 3: Modify the objective function to include penalties for exceeding cover requirements and unmet preferences

    # Penalty variables for labor law violations (not used since checks are omitted)
    penalty_exceed_cover = 1000000  # Penalty for exceeding cover requirement

    # High penalty for unmet shift-off requests on requested day off (to ensure it’s never violated)
    penalty_unmet_off_day = 1000000

    # Define penalties for unmet preferences based on weight
    penalty_unmet_on_weight_3 = 500
    penalty_unmet_off_weight_3 = 400

    penalty_unmet_on_weight_2 = 250
    penalty_unmet_off_weight_2 = 150

    penalty_unmet_on_weight_1 = 50
    penalty_unmet_off_weight_1 = 10

    # Update the objective function to include penalties for exceeding cover and unmet preferences

    model += (
        # Penalize excess cover
        lpSum([
            penalty_exceed_cover * excess_vars[(day_id, shift_id)]
            for day_id in days for shift_id in shift_types
        ]) +

        # Penalize unmet shift-on preferences based on weight
        lpSum(
            penalty_unmet_on_weight_3 * (1 - x[(nurse_id, day_id, shift_id)]) * (shift_on_requests_dict.get((nurse_id, day_id, shift_id), 0) == 3)
            for nurse_id in nurses for day_id in days for shift_id in shift_types
        ) +
        lpSum(
            penalty_unmet_on_weight_2 * (1 - x[(nurse_id, day_id, shift_id)]) * (shift_on_requests_dict.get((nurse_id, day_id, shift_id), 0) == 2)
            for nurse_id in nurses for day_id in days for shift_id in shift_types
        ) +
        lpSum(
            penalty_unmet_on_weight_1 * (1 - x[(nurse_id, day_id, shift_id)]) * (shift_on_requests_dict.get((nurse_id, day_id, shift_id), 0) == 1)
            for nurse_id in nurses for day_id in days for shift_id in shift_types
        ) +

        # Penalize unmet shift-off preferences based on weight
        lpSum(
            penalty_unmet_off_weight_3 * x[(nurse_id, day_id, shift_id)] * (shift_off_requests_dict.get((nurse_id, day_id, shift_id), 0) == 3)
            for nurse_id in nurses for day_id in days for shift_id in shift_types
        ) +
        lpSum(
            penalty_unmet_off_weight_2 * x[(nurse_id, day_id, shift_id)] * (shift_off_requests_dict.get((nurse_id, day_id, shift_id), 0) == 2)
            for nurse_id in nurses for day_id in days for shift_id in shift_types
        ) +
        lpSum(
            penalty_unmet_off_weight_1 * x[(nurse_id, day_id, shift_id)] * (shift_off_requests_dict.get((nurse_id, day_id, shift_id), 0) == 1)
            for nurse_id in nurses for day_id in days for shift_id in shift_types
        ),
        "Total_Penalty_Objective"
    )

    # 3. Maximum Shifts per Nurse
    for nurse_id in nurses:
        max_shifts = max_shifts_dict[nurse_id]
        for shift_id in shift_types:
            model += (
                lpSum(x[(nurse_id, day_id, shift_id)] for day_id in days) <= max_shifts.get(shift_id, 0),
                f"max_shifts_{nurse_id}_{shift_id}"
            )

    # Coverage constraints
    for day_id in days:
        for shift_id in shift_types:
            if (day_id, shift_id) in cover_requirements_dict:
                requirement = cover_requirements_dict[(day_id, shift_id)]
                model += (lpSum(x[(nurse_id, day_id, shift_id)] for nurse_id in nurses) >= requirement, f"min_cover_{day_id}_{shift_id}")

    # 4. Maximum Total Working Time
    # Assuming each shift is 8 hours (480 minutes)
    for nurse_id in nurses:
        model += (
            lpSum(x[(nurse_id, day_id, shift_id)] for day_id in days for shift_id in shift_types) * 480 <= max_total_minutes[nurse_id],
            f"max_total_minutes_{nurse_id}"
        )

    # 5. Consecutive Shifts
    # Ensuring that nurses do not exceed max consecutive shifts
    for nurse_id in nurses:
        max_consecutive = max_consecutive_shifts[nurse_id]
        for day_index in range(len(days) - max_consecutive):
            model += (
                lpSum(x[(nurse_id, days[day_index + d], shift_id)] for d in range(max_consecutive) for shift_id in shift_types) <= max_consecutive,
                f"max_consecutive_shifts_{nurse_id}_{day_index}"
            )

    # Labor Law Constraints
    for nurse_id in nurses:
        for day_id in days:
            if day_id < max(days):
                next_day = day_id + 1
                # Rule 1: A nurse cannot work an Early shift (E) if they worked a Day shift (D) the previous day
                model += (
                    x[(nurse_id, day_id, 'D')] + x[(nurse_id, next_day, 'E')] <= 1,
                    f"no_D_to_E_{nurse_id}_{day_id}"
                )

                # Rule 2: A nurse cannot work a Day shift (D) if they worked a Night shift (L) the previous day
                model += (
                    x[(nurse_id, day_id, 'L')] + x[(nurse_id, next_day, 'D')] <= 1,
                    f"no_L_to_D_{nurse_id}_{day_id}"
                )

                # Rule 3: A nurse cannot work an Early shift (E) if they worked a Night shift (L) the previous day
                model += (
                    x[(nurse_id, day_id, 'L')] + x[(nurse_id, next_day, 'E')] <= 1,
                    f"no_L_to_E_{nurse_id}_{day_id}"
                )

    # Add the constraint that each nurse can work at most one shift per day
    for nurse_id in nurses:
        for day_id in days:
            model += (
                lpSum(x[(nurse_id, day_id, shift_id)] for shift_id in shift_types) <= 1,
                f"one_shift_per_day_{nurse_id}_{day_id}"
            )

    # Hard Constraints: Days Off
    # Nurses cannot be assigned on their requested days off
    for nurse_id in nurses:
        if nurse_id in days_off_dict:
            for day_id in days_off_dict[nurse_id]:
                if day_id in days:
                    for shift_id in shift_types:
                        model += (x[(nurse_id, day_id, shift_id)] == 0, f"day_off_{nurse_id}_{day_id}_{shift_id}")

    # Solve the model
    st.write("Solving the optimisation model. This may take a few moments...")
    model.solve(PULP_CBC_CMD(msg=True, timeLimit=3600, options=["feasibilityTol", "1e-12", "maxIterations", "10000000"]))

    # Extract the solution into a dictionary for easier checks
    assigned_shifts = {(nurse_id, day_id, shift_id): var.varValue for (nurse_id, day_id, shift_id), var in x.items() if var.varValue > 0}

    # Extract solution into a readable format with aggregated nurse lists
    # Create a dictionary to hold the schedule
    schedule_data = []

    total_days = max(days)  # Total number of days to be scheduled
    week_length = 7  # One week has 7 days

    for day_number in days:
        week = (day_number // week_length) + 1  # Calculate the week number starting from 1
        day_of_week = (day_number % week_length) + 1  # Day of the week: 1 to 7

        # Ensure shifts are ordered as E, D, L
        for shift in ['E', 'D', 'L']:
            # Get nurses assigned to this shift on this day
            nurse_ids = [nurse_id for (nurse_id, day_id, shift_id) in assigned_shifts if day_id == day_number and shift_id == shift]
            nurse_ids = ",".join(sorted(set(nurse_ids)))
            schedule_data.append({
                "Week": week,
                "Day": day_of_week,
                "Day_Number": day_number,
                "Shift": shift,
                "Nurse_ID": nurse_ids
            })

    # Now create a DataFrame from schedule_data
    schedule_df = pd.DataFrame(schedule_data)

    # Now, apply the additional optimization functions to further optimize the schedule

    # Define prohibited shifts according to labor law
    prohibited_shifts = {
        ('D', 'E'): "Early shift after Day shift",
        ('L', 'D'): "Day shift after Late shift",
        ('L', 'E'): "Early shift after Late shift"
    }

   # Function definitions (from your provided code)
    # Include all the additional functions provided

    # Function to count the number of weekend shifts for each nurse
    def count_weekend_shifts(schedule_df, nurses, days):
        weekend_days = [5 + 7 * week for week in range((len(days) + 6) // 7)] + \
                       [6 + 7 * week for week in range((len(days) + 6) // 7)]
        nurse_shifts = {nurse: 0 for nurse in nurses}

        # Iterate over the schedule and count weekend shifts
        for _, row in schedule_df.iterrows():
            day_number = row["Day_Number"]
            nurse_ids = row["Nurse_ID"]

            if day_number in weekend_days:
                for nurse in nurse_ids.split(','):
                    if nurse in nurse_shifts:
                        nurse_shifts[nurse] += 1

        return nurse_shifts

    def extract_valid_nurse_ids(schedule_df):
        """Extract valid nurse IDs from the schedule DataFrame."""
        valid_nurses = set()
        for nurse_ids in schedule_df["Nurse_ID"]:
            nurses_list = nurse_ids.split(',')
            for nurse in nurses_list:
                if nurse.strip():  # Avoid empty strings
                    valid_nurses.add(nurse.strip())
        return valid_nurses

    def validate_nurse_ids(nurse_ids, valid_nurses):
        """Check if all nurse IDs are valid and return only valid ones."""
        valid_nurse_list = [nurse.strip() for nurse in nurse_ids.split(',') if nurse.strip() in valid_nurses]
        return valid_nurse_list

    def filter_invalid_nurses_from_schedule(schedule_df, valid_nurses):
        """Filter out invalid nurse IDs from the schedule DataFrame."""
        for idx, row in schedule_df.iterrows():
            nurse_ids = row["Nurse_ID"]
            valid_nurses_only = validate_nurse_ids(nurse_ids, valid_nurses)
            schedule_df.at[idx, "Nurse_ID"] = ','.join(valid_nurses_only)
        return schedule_df

    def revalidate_schedule(schedule_df, valid_nurses):
        """Revalidate the entire schedule to ensure all nurses are valid."""
        return filter_invalid_nurses_from_schedule(schedule_df, valid_nurses)

    def check_labor_law_for_swap_extended(nurse, schedule_df, day, shift, prohibited_shifts):
        """
        Extended labor law check that not only checks the swap day but also adjacent days to ensure no violations.
        """
        nurse_shifts = schedule_df[schedule_df["Nurse_ID"].str.contains(nurse, na=False)]
        nurse_shifts = nurse_shifts.sort_values("Day_Number")
        
        # Check previous day shift
        prev_shift = nurse_shifts[nurse_shifts["Day_Number"] == (day - 1)]
        if not prev_shift.empty:
            prev_shift_type = prev_shift["Shift"].values[0]
            if (prev_shift_type, shift) in prohibited_shifts:
                return True  # Violation for the previous day

        # Check next day shift
        next_shift = nurse_shifts[nurse_shifts["Day_Number"] == (day + 1)]
        if not next_shift.empty:
            next_shift_type = next_shift["Shift"].values[0]
            if (shift, next_shift_type) in prohibited_shifts:
                return True  # Violation for the next day

        return False  # No violation

    def swap_shifts(nurse1, nurse2, day, shift, schedule_df, shift_on_requests_dict, shift_off_requests_dict, valid_nurses, prohibited_shifts):
        """
        Perform a swap between nurse1 and nurse2 for a specific day and shift, considering only weight 3 preferences, labor law constraints, 
        and ensuring that no nurse is assigned to more than one shift per day.
        """
        # Ensure the nurses involved in the swap are valid
        if nurse1 not in valid_nurses or nurse2 not in valid_nurses:
            return False

        # Check if nurse1 or nurse2 are already assigned to a shift on the same day
        nurse1_shifts_on_day = schedule_df.loc[
            (schedule_df["Day_Number"] == day) & 
            (schedule_df["Nurse_ID"].str.contains(nurse1, na=False))
        ]
        
        nurse2_shifts_on_day = schedule_df.loc[
            (schedule_df["Day_Number"] == day) & 
            (schedule_df["Nurse_ID"].str.contains(nurse2, na=False))
        ]

        # If nurse1 or nurse2 are already working a different shift that day, we cannot proceed with the swap
        if not nurse1_shifts_on_day.empty and nurse1_shifts_on_day["Shift"].values[0] != shift:
            return False

        if not nurse2_shifts_on_day.empty and nurse2_shifts_on_day["Shift"].values[0] != shift:
            return False

        # Check for labor law violations before performing the swap
        if check_labor_law_for_swap_extended(nurse1, schedule_df, day, shift, prohibited_shifts) or \
           check_labor_law_for_swap_extended(nurse2, schedule_df, day, shift, prohibited_shifts):
            return False  # No swap due to labor law violation

        # Get rows for the nurses
        nurse1_row = schedule_df.loc[
            (schedule_df["Day_Number"] == day) & 
            (schedule_df["Shift"] == shift) & 
            (schedule_df["Nurse_ID"].str.contains(nurse1, na=False))
        ]
        nurse2_row = schedule_df.loc[
            (schedule_df["Day_Number"] == day) & 
            (schedule_df["Shift"] == shift) & 
            (schedule_df["Nurse_ID"].str.contains(nurse2, na=False))
        ]

        if not nurse1_row.empty and not nurse2_row.empty:
            nurse1_shift = nurse1_row["Nurse_ID"].values[0]
            nurse2_shift = nurse2_row["Nurse_ID"].values[0]

            # Strictly protect weight 3 preferences
            nurse1_preference = shift_on_requests_dict.get((nurse1, day, shift), 0)
            nurse2_preference = shift_on_requests_dict.get((nurse2, day, shift), 0)

            if nurse1_preference == 3 or nurse2_preference == 3:
                return False  # No swap due to preference protection

            # Remove duplicate assignments if the nurses are already assigned to a shift on the same day
            if nurse1 in nurse2_shift or nurse2 in nurse1_shift:
                return False  # Prevent duplicate assignment to the same shift

            # Perform the swap
            schedule_df.loc[nurse1_row.index, "Nurse_ID"] = nurse1_shift.replace(nurse1, nurse2)
            schedule_df.loc[nurse2_row.index, "Nurse_ID"] = nurse2_shift.replace(nurse2, nurse1)

            # Revalidate the schedule after the swap
            schedule_df = revalidate_schedule(schedule_df, valid_nurses)

            return True  # Swap occurred

        return False  # No swap occurred

    def calculate_variance(nurse_shifts):
        """Calculate the variance in weekend shifts for all nurses."""
        weekend_shifts = np.array(list(nurse_shifts.values()))
        return np.var(weekend_shifts)

    def optimize_weekend_shifts(nurse_shifts, schedule_df, shift_on_requests_dict, valid_nurses, prohibited_shifts, num_iterations=30, seed=42):
        """
        Optimize weekend shift distribution by reassigning shifts from the nurse with the most weekend shifts
        to the nurse with the fewest, while considering labor law constraints and protecting preferences.
        """
        np.random.seed(seed)  # Set random seed for reproducibility
        weekend_days = [5 + 7 * week for week in range((len(days) + 6) // 7)] + \
                       [6 + 7 * week for week in range((len(days) + 6) // 7)]
        
        current_variance = calculate_variance(nurse_shifts)
        nurses_list = list(nurse_shifts.keys())

        for _ in range(num_iterations):
            # Identify the nurse with the most and least weekend shifts
            busiest_nurse = max(nurse_shifts, key=nurse_shifts.get)
            least_busy_nurse = min(nurse_shifts, key=nurse_shifts.get)

            # If they have the same number of shifts, no balancing is needed
            if nurse_shifts[busiest_nurse] == nurse_shifts[least_busy_nurse]:
                continue

            # Find a weekend shift assigned to the busiest nurse
            for day in weekend_days:
                for shift in ['E', 'D', 'L']:
                    nurse1_shifts_on_day = schedule_df.loc[
                        (schedule_df["Day_Number"] == day) & 
                        (schedule_df["Nurse_ID"].str.contains(busiest_nurse, na=False)) &
                        (schedule_df["Shift"] == shift)
                    ]
                    
                    if not nurse1_shifts_on_day.empty:
                        # Attempt to swap this shift to the least busy nurse
                        swap_result = swap_shifts(busiest_nurse, least_busy_nurse, day, shift, schedule_df, 
                                                  shift_on_requests_dict, shift_off_requests_dict, valid_nurses, prohibited_shifts)
                        
                        if swap_result:
                            # Recalculate the weekend shift distribution after the swap
                            new_nurse_shifts = count_weekend_shifts(schedule_df, nurses_list, days)
                            new_variance = calculate_variance(new_nurse_shifts)
                            
                            # Accept the swap if variance decreases or stays the same
                            if new_variance <= current_variance:
                                current_variance = new_variance
                                nurse_shifts = new_nurse_shifts
                            else:
                                # Revert swap if variance increases
                                swap_shifts(least_busy_nurse, busiest_nurse, day, shift, schedule_df, 
                                            shift_on_requests_dict, shift_off_requests_dict, valid_nurses, prohibited_shifts)
        
        return nurse_shifts, current_variance

    def finalize_schedule_with_swaps(schedule_df, shift_on_requests_dict, nurses, days):
        """Finalize the schedule with optimized weekend shifts and further optimizations."""
        # Extract valid nurse IDs from the schedule
        valid_nurses = extract_valid_nurse_ids(schedule_df)
        
        # Filter the schedule to remove invalid nurse IDs
        schedule_df = filter_invalid_nurses_from_schedule(schedule_df, valid_nurses)
        
        nurse_shifts = count_weekend_shifts(schedule_df, nurses, days)

        # Optimize the weekend shifts
        optimized_nurse_shifts, final_variance = optimize_weekend_shifts(
            nurse_shifts, schedule_df, shift_on_requests_dict, valid_nurses, prohibited_shifts)

        # Now, apply triplet optimization
        def count_nurse_triplets(schedule_df):
            """Count how often triplets of nurses are assigned to the same shift."""
            nurse_triplet_counts = defaultdict(int)  # Dictionary to store counts of nurse triplets

            # Loop through each row in the schedule
            for _, row in schedule_df.iterrows():
                nurse_ids = row["Nurse_ID"].split(",")  # Get list of nurse IDs assigned to the shift
                nurse_ids = [nurse.strip() for nurse in nurse_ids if nurse.strip()]  # Clean up IDs
                
                # Count all combinations of nurse triplets in the shift
                for nurse_triplet in combinations(nurse_ids, 3):
                    # Sort the triplet to ensure consistency
                    triplet = tuple(sorted(nurse_triplet))
                    nurse_triplet_counts[triplet] += 1

            return nurse_triplet_counts

        nurse_triplet_counts = count_nurse_triplets(schedule_df)

        def identify_frequent_triplets(nurse_triplet_counts, total_shifts, threshold=0.10):
            """Identify triplets that work together for 10% or more of the shifts."""
            frequent_triplets = []
            for triplet, count in nurse_triplet_counts.items():
                if count >= threshold * total_shifts:
                    frequent_triplets.append((triplet, count))
            return frequent_triplets

        def swap_triplet_member(triplet, schedule_df, nurses, shift_on_requests_dict, shift_off_requests_dict, valid_nurses):
            """Attempt to swap one member of the triplet with another nurse who doesn't often work with the other two."""
            nurse1, nurse2, nurse3 = triplet

            # Find a nurse that doesn't frequently work with the other two in the triplet
            other_nurses = [nurse for nurse in nurses if nurse not in triplet]

            # Loop through the other nurses and attempt to swap one nurse from the triplet
            for other_nurse in other_nurses:
                for day in days:
                    for shift in ['E', 'D', 'L']:
                        # Try swapping one member of the triplet
                        for nurse_to_swap in triplet:
                            swap_result = swap_shifts(nurse_to_swap, other_nurse, day, shift, schedule_df, shift_on_requests_dict, shift_off_requests_dict, valid_nurses, prohibited_shifts)
                            if swap_result:
                                return True, nurse_to_swap, other_nurse  # Swap successful
            return False, None, None  # No swap occurred

        def optimize_triplet_swaps(schedule_df, nurse_triplet_counts, nurses, shift_on_requests_dict, shift_off_requests_dict, total_shifts, valid_nurses):
            """Optimize the schedule by swapping members of frequent triplets (working together more than 10% of the shifts)."""
            frequent_triplets = identify_frequent_triplets(nurse_triplet_counts, total_shifts)
            if not frequent_triplets:
                return schedule_df

            # Try to swap a member of each frequent triplet
            for triplet, count in frequent_triplets:
                swap_result, swapped_out_nurse, swapped_in_nurse = swap_triplet_member(triplet, schedule_df, nurses, shift_on_requests_dict, shift_off_requests_dict, valid_nurses)
                if swap_result:
                    # Update the triplet counts after the swap
                    nurse_triplet_counts = count_nurse_triplets(schedule_df)
                else:
                    pass  # No swap made for this triplet
            
            return schedule_df

        total_shifts = len(schedule_df)  # Total number of shifts

        # Optimize triplets
        schedule_df = optimize_triplet_swaps(schedule_df, nurse_triplet_counts, nurses, shift_on_requests_dict, shift_off_requests_dict, total_shifts, valid_nurses)

        return schedule_df

    # Finalize and optimize the schedule
    optimized_schedule_df = finalize_schedule_with_swaps(schedule_df, shift_on_requests_dict, nurses, days)

    # Display the optimized schedule
    st.success("Optimisation completed. Here is the optimised schedule in csv format available for donwload:")
    st.dataframe(optimized_schedule_df)

    # Optionally, provide a download button for the schedule
    csv = optimized_schedule_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Optimised Schedule as CSV",
        data=csv,
        file_name='optimized_nurse_schedule.csv',
        mime='text/csv',
    )

    # Now, include the calendar view
    st.header("Calendar View of Schedule")

    # Prepare events for the calendar component
    events = []
    shift_labels = {'E': 'Early shift', 'D': 'Day shift', 'L': 'Late shift'}

    # Assume day 0 is 7 days from today
    start_date = datetime.today() + timedelta(days=7)

    # Generate 'schedule_aggregated' from 'optimized_schedule_df'
    schedule_aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, row in optimized_schedule_df.iterrows():
        week_number = int(row['Week'])
        day_of_week = int(row['Day'])
        shift = row['Shift']
        nurse_ids = row['Nurse_ID'].split(',')
        nurse_ids = [nurse.strip() for nurse in nurse_ids if nurse.strip()]
        schedule_aggregated[week_number][day_of_week][shift].extend(nurse_ids)

    # Now, prepare events for the calendar component
    for week_number in schedule_aggregated:
        for day_of_week in schedule_aggregated[week_number]:
            # Calculate the shift_day
            shift_day = start_date + timedelta(days=((week_number - 1) * 7 + (day_of_week - 1)))  # day_of_week starts from 1
            for shift, nurses in schedule_aggregated[week_number][day_of_week].items():
                for nurse in nurses:
                    if shift == 'E':
                        start_time = "00:00:00"
                        end_time = "08:00:00"
                    elif shift == 'D':
                        start_time = "08:00:00"
                        end_time = "16:00:00"
                    elif shift == 'L':
                        start_time = "16:00:00"
                        end_time = "23:59:59"

                    event = {
                        "title": f"Nurse {nurse} - {shift_labels.get(shift, shift)}",
                        "start": f"{shift_day.strftime('%Y-%m-%d')}T{start_time}",
                        "end": f"{shift_day.strftime('%Y-%m-%d')}T{end_time}",
                    }
                    events.append(event)

    calendar_options = {
        "editable": False,
        "selectable": True,
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,timeGridDay",
        },
        "initialView": "dayGridMonth",
    }

    custom_css = """
        .fc-event-time {
            font-style: italic;
        }
        .fc-event-title {
            font-weight: bold;
        }
    """

    # Use the Calendar component to display the schedule
    calendar_component = calendar(events=events, options=calendar_options, custom_css=custom_css)
    st.write(calendar_component)

else:
    st.warning("Please upload all the required CSV files to proceed.")


# In[ ]:




