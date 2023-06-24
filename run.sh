# This script implements the replictation codes
echo "* Implementing Debiased-Lasso.....\n"
Rscript R/DebiasedLasso.R

echo "* Implmenting Lasso-BLP.....\n"
# 1. Data-simulation in R and C++
echo "* 1. Simulating datasets.....\n"
echo "* 1.1. Simulating K = 100.....\n"
Rscript R/BLP_data_simulation_k100.R
echo "* 1.2. Simulating K = 200.....\n"
Rscript R/BLP_data_simulation_k200.R
echo "* 1.3. Simulating true dataset.....\n"
Rscript R/BLP_data_simiulation_true.R

# 2. MPEC Lasso-BLP estimation with Python
echo "* 2. Lasso-BLP estimation with MPEC.....\n"
echo "* 2.1. Post-Single Lasso-BLP K = 100.....\n"
python3 Python/BLP_single_K100.py
echo "* 2.2. Debiased Lasso-BLP K = 100.....\n"
python3 Python/BLP_debiased_K100.py
echo "* 2.3. Post-Single Lasso-BLP K = 200.....\n"
python3 Python/BLP_single_K200.py
echo "* 2.4. Debiased Lasso-BLP K = 200.....\n"
python3 Python/BLP_debiased_K200.py
echo "* 2.5. True BLP model.....\n"
pytho3 Python/BLP_true.py

# 3. Visualization of results with R
echo "* 3. Visualizing the results.....\n"
Rscript R/DebiasedBLP.R