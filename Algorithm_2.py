
import logging
from pyscipopt import Model, quicksum
import numpy as np
import sys
import random

logging.basicConfig(filename='ilp_process4n.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_lp_problem(file_path, active_indices):
    model = Model("ILP Problem")
    try:
        model.readProblem(file_path)
        all_vars = model.getVars()
        if any(i >= len(all_vars) or i < 0 for i in active_indices):
            raise IndexError("One or more indices are out of the valid range.")
        
        active_vars = [all_vars[i] for i in active_indices]
        non_active_vars = [var for i, var in enumerate(all_vars) if i not in active_indices]
        
   

        logging.info("Objective function redefined with active variables first.")
    except Exception as e:
        logging.error(f"Error while loading and preparing model: {e}")
        return None
    return model





def get_user_input(prompt):
    input_str = input(prompt)
    try:
        return list(map(int, input_str.split()))
    except ValueError:
        logging.error("Invalid input. Please enter valid integer indices separated by spaces.")
        return None



def identify_active_variables(model):
    active_indices = get_user_input("Enter the indices of active variables separated by spaces: ")
    if active_indices is not None and len(active_indices) > 0:
        all_vars = model.getVars()
        if all(index < len(all_vars) for index in active_indices):
            active_vars = [all_vars[i] for i in active_indices]
            k = len(active_vars)  
            logging.info(f"Identified {k} active variables based on user input.")
            return active_vars, active_indices, k
        else:
            logging.error(f"Input error: Some indices exceed the number of variables in the model ({len(all_vars)}).")
            return [], [], 0
    else:
        logging.error("No valid indices of active variables entered.")
        return [], [], 0

def vu_values(k):
    v = [[np.cos(2 * np.pi * i * j / k) for i in range(k)] for j in range((k - 1) // 2 + 1)]
    u = [[np.sin(2 * np.pi * i * j / k) for i in range(k)] for j in range((k - 1) // 2 + 1)]
    logging.info(f"Generated v and u matrices for k = {k}, v = {v}, u = {u}")
    return v, u

def create_pv_pu(model, v, u, active_vars, j):
    logging.info(f"Creating projection vectors for component {j}")
    p_v = quicksum(v[j][i] * active_vars[i] for i in range(len(active_vars)))
    p_u = quicksum(u[j][i] * active_vars[i] for i in range(len(active_vars)))
    return p_v, p_u


def creating_P0(model, active_vars):
    k = len(active_vars)
    v, u = vu_values(k)
    M = 10000
    r = [model.addVar(vtype="BINARY", name=f"r_{j}") for j in range((k - 1) // 2 + 1)]
    for j in range(((k - 1) // 2) + 1):
        p_v, p_u = create_pv_pu(model, v, u, active_vars, j)

        if j == 0 or (k % 2 == 0 and j == k // 2):
            model.addCons(p_v <= r[j] * M)
            model.addCons(p_v >= -r[j] * M)
        else:
            p_j = (p_v * p_v + p_u * p_u)
            model.addCons(p_j <= r[j] * M)

        logging.info(f'Added constraints for variable set {j}')
        
    model.addCons(quicksum(r[j] for j in range((k - 1) // 2 + 1)) <= k // 2)
    output_lp_path = "P_0-const.lp"
    model.writeProblem(output_lp_path)
    logging.info(f"P0 constraints written to {output_lp_path}")
    model.optimize()
    if model.getStatus() == "optimal":
        logging.info(f'P0 solved optimally with objective: {model.getObjVal()}')
        return model.getObjVal()
    else:
        logging.warning('P0 optimization was infeasible')
        return -np.inf



def create_essential_and_projected_sets(k, n, i):
    base_value = i // k
    remainder = i % k
    ucp = [base_value] * k
    for _ in range(remainder):
        index = random.choice(range(k))
        ucp[index] += 1

    atom_points = []
    #for _ in range(2):
     #   indices = random.sample(range(k), 2)
      #  if ucp[indices[0]] > 0:
       #     new_point = ucp.copy()
        #    new_point[indices[0]] += 1
         #   new_point[indices[1]] -= 1
          #  atom_points.append(new_point)
    
    essential_set = [pt + [0] * (n - k) for pt in [ucp] + atom_points]
    projected_set = essential_set
    logging.info(f'Projected set: {projected_set}')
    return essential_set, projected_set


def creating_subproblems_L_i(file_path, k, i, active_indices):
    logging.info(f"Setting up subproblem L_{i}")
    L_i_model = load_lp_problem(file_path, active_indices) 
    if L_i_model is None:
        logging.error(f"Failed to load the model for subproblem L_{i}.")
        return None
   
    q_i = L_i_model.addVar(vtype="INTEGER", name=f"q_{i}")
    x = L_i_model.getVars()[:k]
    sum_constraint = quicksum(x[j] for j in range(k)) == q_i * k + i
    L_i_model.addCons(sum_constraint)
    logging.info(f"Subproblem L_{i} set up with new constraint added.")

    return L_i_model



def add_feasibility_constraints(model, projected_set, layer_index):
    x = model.getVars()  

    for point_index, vec in enumerate(projected_set):
        logging.info(f"Adding constraints for vector {point_index} at layer {layer_index}: {vec}")
        
        if not all(isinstance(item, (int, float)) for item in vec):
            logging.error(f"Vector {vec} contains non-numeric elements.")
            continue
        
        base_index = random.randint(0, len(vec) - 1)
        base_var = x[base_index]
       
        for i in range(len(vec)):
            if i != base_index:
                difference = vec[i] - vec[base_index]
                model.addCons(x[i] == base_var + difference)
                logging.info(f"Added constraint: x[{i}] = x[{base_index}] + {difference}")

    logging.info(f"Constraints added for layer {layer_index}")



def compute_T_k(i, k, model):
    T_k_values = []
    x = model.getVars()
    active_indices = range(k)
    
    for m in range(1, (k-1)//2 + 1):  
        cos_m = [np.cos(2 * np.pi * m * j / k) for j in active_indices]
        sin_m = [np.sin(2 * np.pi * m * j / k) for j in active_indices]

        p_v = quicksum(cos_m[j] * x[j] for j in active_indices)
        p_u = quicksum(sin_m[j] * x[j] for j in active_indices)
        p_j_squared = p_v * p_v + p_u * p_u
        
        model.addCons(p_j_squared >= 1e-6)
        logging.info(f"Constraint 16: {p_j_squared >= 1e-6}")
        
        for shift in range(k):
            shifted_cos_m = np.roll(cos_m, -shift)
            p_sigma = quicksum(shifted_cos_m[j] * x[j] for j in active_indices)
            T_k_value = 2 * p_sigma / p_j_squared
            T_k_values.append(T_k_value)
         
    
    with open('T_k_values_output.txt', 'a') as file:
        file.write(f'T_k values for k = {k}: {T_k_values}\n')
        logging.info(f'T_k values is written in T_k_values_output.txt')

    logging.info(f'T_k values computed = {T_k_values}')
    return T_k_values



def add_constraint_13(model, projected_set, T_values, k):
    for vec in projected_set:
        constraint_expression = quicksum(vec[j] * T_values[j] for j in range(k)) + 1
        model.addCons(constraint_expression <= 1e-6)
        logging.info(f"Added constraint for vector {vec} with T-values.")



    
    
    
def main_execution_loop():
    if len(sys.argv) < 2:
        logging.error("No LP file path provided.")
        return
    
    file_path = sys.argv[1]
    active_indices = get_user_input("Enter the indices of active variables separated by spaces: ")
    if active_indices is None or not active_indices:
        logging.error("No valid active indices provided.")
        return

    model1 = load_lp_problem(file_path, active_indices)
    f_star = -np.inf
    if model1:
        logging.info("Loaded initial LP problem successfully.")
        active_vars = [model1.getVars()[index] for index in active_indices if index < len(model1.getVars())]
        k = len(active_vars)
        n = len(model1.getVars())
        if active_vars and k > 0:
            logging.info(f"{k} active variables identified.")
            f_star = creating_P0(model1, active_vars)
            logging.info(f"Result of P0 optimization (f_star initial): {f_star}")

            results_from_feasibility = []
            for i in range(1, k + 1):
                essential_set, projected_set = create_essential_and_projected_sets(k, n, i)
                for vec_index, vec in enumerate(projected_set):
                    L_i_model = creating_subproblems_L_i(file_path, k, i, active_indices)  
                    if L_i_model is None:
                        logging.error(f"Failed to recreate subproblem L_{i} for vector {vec_index}")
                        continue

                    add_feasibility_constraints(L_i_model, [vec], i)
                    L_i_model.optimize()
                    L_i_model.writeProblem(f"L_{i}_vector_{vec_index}_feasibility.lp")
                    logging.info(f"Subproblem L_{i} feasibility written to L_{i}_vector_{vec_index}_feasibility.lp")
                    if L_i_model.getStatus() == "optimal":
                        result = L_i_model.getObjVal()
                        results_from_feasibility.append(result)
                        logging.info(f"Optimized L_{i} with vector {vec}: Feasibility result {result}")
                    else:
                        logging.warning(f"Optimization of L_{i} with vector {vec} was infeasible after adding feasibility constraints")
                    L_i_model.freeTransform()

                if results_from_feasibility:
                    result10 = max(results_from_feasibility)
                    f_star = max(f_star, result10)
                    logging.info(f"Updated f_star after feasibility constraints: {f_star}")

                L_i_model = creating_subproblems_L_i(file_path, k, i, active_indices)
                if L_i_model is None:
                    logging.error(f"Failed to recreate subproblem L_{i} for adding constraint 13")
                    continue

                T_values = compute_T_k(i, k, L_i_model)
                add_constraint_13(L_i_model, projected_set, T_values, k)
                result_path = f"subproblem_L_{i}_optimized_after_constraint13.cip"
                L_i_model.writeProblem(result_path)
                logging.info(f"Subproblem L_{i} after applying constraint 13 written to subproblem_L_{i}_optimized_after_constraint13.cip")

                L_i_model.optimize()
                if L_i_model.getStatus() == "optimal":
                    result20 = L_i_model.getObjVal()
                    logging.info(f"Result after constraint 13: {result20}")
                    f_star = max(f_star, result20)
                    logging.info(f"Updated f_star after constraint 13: {f_star}")
                else:
                    logging.warning(f"Optimization of subproblem L_{i} after constraint 13 was infeasible")
                L_i_model.freeTransform()

        else:
            logging.info("No active variables found.")
    else:
        logging.error("Failed to load initial LP problem.")

    logging.info("Optimization process completed. Final f_star: {}".format(f_star))
    print(f"Final_Result_Alg2: {f_star}")
    return f_star



if __name__ == "__main__":
    main_execution_loop()
