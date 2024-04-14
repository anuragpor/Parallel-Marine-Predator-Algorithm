#include <bits/stdc++.h>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#include <mpi.h>
using namespace std;
#define trace1(x)                cerr<<#x<<": "<<x<<endl
#define trace2(x, y)             cerr<<#x<<": "<<x<<" | "<<#y<<": "<<y<<endl
#define trace3(x, y, z)          cerr<<#x<<":" <<x<<" | "<<#y<<": "<<y<<" | "<<#z<<": "<<z<<endl
#define trace4(a, b, c, d)       cerr<<#a<<": "<<a<<" | "<<#b<<": "<<b<<" | "<<#c<<": "<<c<<" | "<<#d<<": "<<d<<endl
#define trace5(a, b, c, d, e)    cerr<<#a<<": "<<a<<" | "<<#b<<": "<<b<<" | "<<#c<<": "<<c<<" | "<<#d<<": "<<d<<" | "<<#e<< ": "<<e<<endl
#define trace6(a, b, c, d, e, f) cerr<<#a<<": "<<a<<" | "<<#b<<": "<<b<<" | "<<#c<<": "<<c<<" | "<<#d<<": "<<d<<" | "<<#e<< ": "<<e<<" | "<<#f<<": "<<f<<endl


void Debug2D(vector<vector<double>> v)
{
	cout<<"-----------------------------------"<<endl;
	for(auto a : v)
	{
		for(auto b: a)
			cout<<b<<" ";
		cout<<endl;
	}
	cout<<endl;
}

void Debug1D(vector<double> v)
{
	cout<<"-----------------------------------"<<endl;
	for(auto a : v)
	{
		cout<<a<<" ";
	}
	cout<<endl;
}

// Function to initialize the first population of search agents
vector<vector<double>> initialization(int SearchAgents_no, int dim, vector<double>& ub, vector<double>& lb, int rank) {
    int Boundary_no = ub.size(); // Number of boundaries

    // Initialize random seed
    srand((rank+1) * (rank+1));

    vector<vector<double>> Positions(SearchAgents_no, vector<double>(dim));

    // If the boundaries of all variables are equal
    if (Boundary_no == 1) {
        double ub_value = ub[0];
        double lb_value = lb[0];
        for (int i = 0; i < SearchAgents_no; ++i) {
            for (int j = 0; j < dim; ++j) {
                Positions[i][j] = (static_cast<double>(rand()) / RAND_MAX) * (ub_value - lb_value) + lb_value;
            }
        }
    }

    // If each variable has a different lb and ub
    if (Boundary_no > 1) {
        for (int i = 0; i < dim; ++i) {
            double ub_value = ub[i];
            double lb_value = lb[i];
            for (int j = 0; j < SearchAgents_no; ++j) {
                Positions[j][i] = (static_cast<double>(rand()) / RAND_MAX) * (ub_value - lb_value) + lb_value;
            }
        }
    }
    return Positions;
}

// Function to generate Levy steps
vector<vector<double>> levy(int n, int m, double beta, int Iter) {
    // Constants for Levy distribution calculation
    double num = tgamma(1 + beta) * sin(M_PI * beta / 2);
    double den = tgamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2);
    double sigma_u = pow(num / den, 1 / beta);
    srand(static_cast<unsigned int>(std::time(nullptr)));
    // Initialize random number generators
    random_device rd;
    mt19937 gen(rd());
    // mt19937 gen(Iter); // change it
    normal_distribution<double> normal_dist(0.0, sigma_u);

    // Generate random numbers according to Levy distribution
    vector<vector<double>> z(n, vector<double>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double u = normal_dist(gen);
            double v = normal_dist(gen);
            z[i][j] = (u * 0.05) / pow(abs(v), 1 / beta);
        }
    }
    return z;
}

// Example objective functions
double F1(vector<double> x) {
    double sum = 0.0;
    for (auto val : x) {
        sum += val * val;
    }
    return sum;
}

double F2(std::vector<double> x) {
    double sum = 0.0, prod = 1.0;
    for(int i = 0; i < x.size(); i++) {
        sum += std::abs(x[i]);
        prod *= std::abs(x[i]);
    }
    return sum + prod;
}

// Objective function F8
double F8(const std::vector<double> x) {
    int dim = x.size();
    double sum = 0.0;

    for (int i = 0; i < dim; ++i) {
        sum += -x[i] * std::sin(std::sqrt(std::abs(x[i])));
    }
    return sum;
}

// F9
double F9(const std::vector<double> x) {
    int dim = x.size();
    double sum = 0.0;

    for (int i = 0; i < dim; ++i) {
        sum += std::pow(x[i], 2) - 10 * std::cos(2 * M_PI * x[i]);
    }
    return sum + 10 * dim;
}


double Ufun(double x, double a, double k, double m) {
    if (x > a) return k * std::pow(x - a, m);
    else if (x < -a) return k * std::pow(-x - a, m);
    else return 0;
}


// -50, 50, 10
double F12(std::vector<double> x) {
    int dim = x.size();
    double sum1 = 0.0, sum2 = 0.0;
    for(int i = 0; i < dim - 1; i++) {
        sum1 += std::pow((x[i] + 1) / 4, 2) * (1 + 10 * std::pow(std::sin(M_PI * (1 + (x[i + 1] + 1) / 4)), 2));
        sum2 += Ufun(x[i], 10, 100, 4);
    }
    double o = (M_PI / dim) * (10 * std::pow(std::sin(M_PI * (1 + (x[0] + 1) / 4)), 2) + sum1 + std::pow((x[dim - 1] + 1) / 4, 2)) + sum2;
    return o;
}

double F20(std::vector<double> x) {
    std::vector<std::vector<double>> aH = {{10, 3, 17, 3.5, 1.7, 8}, {.05, 10, 17, .1, 8, 14}, {3, 3.5, 1.7, 10, 17, 8}, {17, 8, .05, 10, .1, 14}};
    std::vector<double> cH = {1, 1.2, 3, 3.2};
    std::vector<std::vector<double>> pH = {{.1312, .1696, .5569, .0124, .8283, .5886}, {.2329, .4135, .8307, .3736, .1004, .9991}, {.2348, .1415, .3522, .2883, .3047, .6650}, {.4047, .8828, .8732, .5743, .1091, .0381}};

    double o = 0.0;
    for(int i = 0; i < 4; i++) {
        double sum = 0.0;
        for(int j = 0; j < x.size(); j++) {
            sum += aH[i][j] * std::pow(x[j] - pH[i][j], 2);
        }
        o -= cH[i] * std::exp(-sum);
    }
    return o;
}

double F23(std::vector<double> x) {
    std::vector<std::vector<double>> aSH = {{4, 4, 4, 4}, {1, 1, 1, 1}, {8, 8, 8, 8}, {6, 6, 6, 6}, {3, 7, 3, 7}, {2, 9, 2, 9}, {5, 5, 3, 3}, {8, 1, 8, 1}, {6, 2, 6, 2}, {7, 3.6, 7, 3.6}};
    std::vector<double> cSH = {.1, .2, .2, .4, .4, .6, .3, .7, .5, .5};
    
    double o = 0.0;
    for(int i = 0; i < 10; i++) {
        double sum = 0.0;
        for(int j = 0; j < x.size(); j++) {
            sum += std::pow(x[j] - aSH[i][j], 2);
        }
        o -= std::pow(sum + cSH[i], -1);
    }
    return o;
}

// Objective function type
typedef double (*ObjectiveFunction)(vector<double>);

// Map to store function pointers for different objective functions
map<string, ObjectiveFunction> objectiveFunctions = {
    {"F1", &F1},
    {"F2", &F2},
    {"F8", &F8},
    {"F9", &F9},
    {"F12", &F12},
    {"F20", &F20},
    {"F23", &F23},
};

// Function to perform the Marine Predators Algorithm (MPA)
void MarinePredatorsAlgorithm(int SearchAgents_no, int dim, vector<double>& ub, vector<double>& lb, int Max_iter, double CF, double FADs, double P, int rank, int world_size, string objectiveFunctionName, double& Top_predator_fit, vector<double>& Top_predator_pos) {
    // Initialize search agents
    vector<vector<double>> Prey = initialization(SearchAgents_no, dim, ub, lb, rank);
    // MPA algorithm
    Top_predator_pos.resize(dim);
    Top_predator_fit = numeric_limits<double>::infinity();
    vector<double> Convergence_curve(Max_iter);
    srand((rank+1) * (rank+1));
    // Variables for memory saving
    vector<vector<double>> Prey_old = Prey;
    vector<double> fit_old(SearchAgents_no, numeric_limits<double>::infinity());
    
    // Initialize Xmin and Xmax matrices
    std::vector<std::vector<double>> Xmin(SearchAgents_no, std::vector<double>(dim, lb[0]));
    std::vector<std::vector<double>> Xmax(SearchAgents_no, std::vector<double>(dim, ub[0]));

    // Get the objective function pointer based on the input function name
    ObjectiveFunction fobj = objectiveFunctions[objectiveFunctionName];
    if (fobj == nullptr) {
        cerr << "Error: Objective function '" << objectiveFunctionName << "' not found!" << endl;
        return;
    }

    // Open a text file to write the convergence data
    string filename = objectiveFunctionName + "_MaxIter_" + to_string((int)Max_iter) + "_Dim_" + to_string((int)dim) + "_LB_" + to_string((int)lb[0]) + "_UB_" + to_string((int)ub[0]) + ".csv";
    
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error opening output file!" << endl;
        return;
    }
    
    // Write headers to the file
    outfile << "Function Name,Max Iteration,Dimension,Lower Bounds,Upper Bounds" << endl;
    outfile << objectiveFunctionName << "," << Max_iter << "," << dim << "," << lb[0] << "," << ub[0] << endl;

    // Main loop
    int Iter = 0;
    while (Iter < Max_iter) {
        vector<double> Fitness;
        // Fitness evaluation and update top predator
        for (int i = 0; i < SearchAgents_no; ++i) {
            // Update Prey within bounds
            for (int j = 0; j < dim; ++j) {
                Prey[i][j] = max(min(Prey[i][j], ub[j]), lb[j]);
            }
            double fitness = fobj(Prey[i]);
            Fitness.push_back(fitness);
            if (fitness < Top_predator_fit) {
                Top_predator_fit = fitness;
                Top_predator_pos = Prey[i];
            }
        }
        
        // Marine memory saving
        if(Iter == 0)
        {
            fit_old = Fitness;
            Prey_old = Prey;
        }
        for (int i = 0; i < SearchAgents_no; ++i) {
            if (fit_old[i] < fobj(Prey[i])) {
                Prey[i] = Prey_old[i];
                Fitness[i] = fit_old[i];
            }
        }

        Prey_old = Prey;
        fit_old = Fitness;

        // Foraging behavior simulation
        vector<vector<double>> Elite(SearchAgents_no, Top_predator_pos); // Eq. 10
        CF = pow(1 - (Iter*1.0) / (1.0*Max_iter), (2.0 * Iter) / Max_iter);        // Eq. 9
        
        vector<vector<double>> RL = levy(SearchAgents_no, dim, 1.5, Iter); // Levy random number vector
        vector<vector<double>> RB(SearchAgents_no, vector<double>(dim)); // Brownian random number vector
        
        default_random_engine generator;
        normal_distribution<double> distribution(0.0, 1.0); // Mean = 0, Standard Deviation = 1
        
        for (int i = 0; i < SearchAgents_no; ++i) {
            for (int j = 0; j < dim; ++j) {
                RB[i][j] = distribution(generator);
            }
        }
        P = 0.5;
        vector<vector<double>> stepsize(SearchAgents_no, vector<double>(dim));
        for (int i = 0; i < SearchAgents_no; ++i) {
            for (int j = 0; j < dim; ++j) {
                double R = (static_cast<double>(rand()) / RAND_MAX);
                // Phase 1 (Eq.12)
                if (Iter < (Max_iter*1.0) / 3) {
                    stepsize[i][j] = RB[i][j] * (Elite[i][j] - RB[i][j] * Prey[i][j]);
                    Prey[i][j] += P * R * stepsize[i][j];
                }
                // Phase 2 (Eqs. 13 & 14)
                else if (Iter*3 > Max_iter && Iter*3 < (2 * Max_iter)) {
                    if (i*2 > SearchAgents_no) {
                        stepsize[i][j] = RB[i][j] * (RB[i][j] * Elite[i][j] - Prey[i][j]);
                        Prey[i][j] = Elite[i][j] + P * CF * stepsize[i][j];
                    } else {
                        stepsize[i][j] = RL[i][j] * (Elite[i][j] - RL[i][j] * Prey[i][j]);
                        Prey[i][j] += P * R * stepsize[i][j];
                    }
                }
                // Phase 3 (Eq. 15)
                else {
                    stepsize[i][j] = RL[i][j] * (RL[i][j] * Elite[i][j] - Prey[i][j]);
                    Prey[i][j] = Elite[i][j] + P * CF * stepsize[i][j];
                }
            }
        }
        for (int i = 0; i < SearchAgents_no; ++i) {
            // Update Prey within bounds
            for (int j = 0; j < dim; ++j) {
                Prey[i][j] = max(min(Prey[i][j], ub[j]), lb[j]);
            }

            double fitness = fobj(Prey[i]);
            Fitness[i] = fitness;
            if (fitness < Top_predator_fit) {
                Top_predator_fit = fitness;
                Top_predator_pos = Prey[i];
            }
        }
        // Marine memory saving
        if(Iter == 0)
        {
            fit_old = Fitness;
            Prey_old = Prey;
        }
        for (int i = 0; i < SearchAgents_no; ++i) {
            if (fit_old[i] < Fitness[i]) {
                Prey[i] = Prey_old[i];
                Fitness[i] = fit_old[i];
            }
        }
        fit_old = Fitness;
        Prey_old = Prey;

        std::random_device rd;
	    std::mt19937 gen(rd());
	    std::uniform_real_distribution<double> dist(0.0, 1.0);
	    // Eddy formation and FADs' effect
	    if (dist(gen) < FADs) {
	        std::vector<std::vector<double>> U(SearchAgents_no, std::vector<double>(dim));
	        for (int i = 0; i < SearchAgents_no; ++i) {
	            for (int j = 0; j < dim; ++j) {
	                U[i][j] = dist(gen) < FADs ? 1.0 : 0.0;
	            }
	        }
	        for (int i = 0; i < SearchAgents_no; ++i) {
	            for (int j = 0; j < dim; ++j) {
	                Prey[i][j] += CF * ((Xmin[i][j] + dist(gen) * (Xmax[i][j] - Xmin[i][j])) * U[i][j]);
	            }
	        }
	    } else {
	        double r = dist(gen);
	        int Rs = Prey.size();
	        for (int i = 0; i < SearchAgents_no; ++i) {
	            std::shuffle(Prey.begin(), Prey.end(), gen);
	            std::vector<double> stepsize(dim);
	            for (int j = 0; j < dim; ++j) {
	                stepsize[j] = (FADs * (1 - r) + r) * (Prey[i][j] - Prey[i][j]);
	            }
	            for (int j = 0; j < dim; ++j) {
	                Prey[i][j] += stepsize[j];
	            }
	        }
	    }
	    outfile << Iter << "," << Top_predator_fit << endl;

        // Logic to shuffle Prey between processors based on specific iterations
        if (Iter % 10 == 0 && Iter != 0) {
            auto p1 = Prey;
            // Flatten the 2D vector into a 1D array
            std::vector<double> flat_Prey(SearchAgents_no * dim);
            for (int i = 0; i < SearchAgents_no; ++i) {
                for (int j = 0; j < dim; ++j) {
                    flat_Prey[i * dim + j] = Prey[i][j];
                }
            }

            // Gather the Prey arrays from all processes to the root process
            std::vector<double> flat_AllPrey;

            if (rank == 0) {
                flat_AllPrey.resize(SearchAgents_no * dim * world_size); // Resize AllPrey on the root process
            }

            MPI_Gather(flat_Prey.data(), SearchAgents_no * dim, MPI_DOUBLE, flat_AllPrey.data(), SearchAgents_no * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            vector<vector<double>> AllPrey;
            if (rank == 0) {
                AllPrey.resize(world_size * SearchAgents_no, std::vector<double>(dim));
                for (int i = 0; i < world_size * SearchAgents_no; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        AllPrey[i][j] = flat_AllPrey[i * dim + j];
                    }
                } 
                std::random_device rd;
                std::mt19937 gen(rd());
                std::shuffle(AllPrey.begin(), AllPrey.end(), gen);      

                // // Flatten the shuffled AllPrey vector back into a 1D array
                for (int i = 0; i < world_size * SearchAgents_no; ++i) {
                    for (int j = 0; j < dim; ++j) {
                        flat_AllPrey[i * dim + j] = AllPrey[i][j];
                    }
                }
            }

            // Scatter the shuffled AllPrey vector back to each process
            MPI_Scatter(flat_AllPrey.data(), SearchAgents_no * dim, MPI_DOUBLE, flat_Prey.data(), SearchAgents_no * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Reshape the 1D array back into a 2D vector on each process
            for (int i = 0; i < SearchAgents_no; ++i) {
                for (int j = 0; j < dim; ++j) {
                    Prey[i][j] = flat_Prey[i * dim + j];
                }
            }
        }      
        // Increment iteration count
        Iter++;
        Convergence_curve[Iter] = Top_predator_fit;
    }
    outfile.close();
}

double global_min_fitness;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize MPI
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the total number of processes

    // Parameters for MPA algorithm
    int SearchAgents_no = 2000; // Number of search agents
    int dim = 50;              // Dimension of the problem
    vector<double> ub(dim, 50); // Upper bounds
    vector<double> lb(dim, -50);    // Lower bounds
    int Max_iter = 500;      // Maximum number of iterations
    double CF = 0.5;          // Constant factor for Phase 2
    double FADs = 0.2;        // FADs effect probability
    double P = 0.5;           // Constant factor for Phase 1

    string objectiveFunctionName = "F1";

    // Divide the workload among processes
    int chunk_size = SearchAgents_no / world_size;
    int remainder = SearchAgents_no % world_size;
    int start_index = world_rank * chunk_size;
    int end_index = start_index + chunk_size;

    // Adjust the workload for the last process
    if (world_rank == world_size - 1) {
        end_index += remainder;
    }

    // Start the timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    vector<double> Top_predator_pos;
    double Top_predator_fit;
    // Perform MPA only for the assigned portion of search agents
    MarinePredatorsAlgorithm(end_index - start_index, dim, ub, lb, Max_iter, CF, FADs, P, world_rank, world_size, objectiveFunctionName, Top_predator_fit, Top_predator_pos);

    // Reduce the minimum fitness value across all processors
    double local_min_fitness = Top_predator_fit;
    MPI_Reduce(&local_min_fitness, &global_min_fitness, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Stop the timer and calculate the elapsed time
    std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    double local_time = time_span.count();

    // Reduce the maximum execution time across all processors
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print the maximum execution time and global minimum fitness on the root process
    if (world_rank == 0) {
        std::cout << "Maximum execution time: " << max_time << " seconds.\n";
        std::cout << "Global Minimum Fitness: " << global_min_fitness << endl;
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}