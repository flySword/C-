#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <cassert>
#include "block_cyclic_mat.h"
#include "scalapack.h"

static double pdgesvd_flops(blas_idx_t M, blas_idx_t N)
{
    // From: 
    // http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=921
    return (14.0 * M * N * N + 8.0 * N * N * N)/1024.0/1024.0/1024.0;
}

// 
// Compute ||A||
//
static double one_norm(std::shared_ptr<block_cyclic_mat_t> matrix)
{    
    auto norm = '1';        
    auto m    = matrix->global_rows();
    auto n    = matrix->global_cols();
    auto ia   = static_cast<blas_idx_t>(1);
    auto ja   = static_cast<blas_idx_t>(1);

    std::vector<double> work(matrix->local_cols());
    return pdlange_(norm, m, n, matrix->local_data(), ia, ja, matrix->descriptor(), work.data());
}

//
// Compute ||U'*U - I|| or ||V' * V - I||
//
static double compute_residual(std::shared_ptr<block_cyclic_mat_t> matrix, bool isU)
{    
    auto min_mn     =  std::min(matrix -> global_rows(), matrix -> global_cols());
    auto identity   = block_cyclic_mat_t::diagonal(matrix -> grid(), min_mn, min_mn);
    auto normal     = 'N';
    auto transposed = 'T';
    auto ia         = static_cast<blas_idx_t>(1);
    auto ja         = static_cast<blas_idx_t>(1);
    auto alpha      = 1.0;
    auto beta       = -1.0;

    if (isU)
    {
        auto m_global = matrix -> global_rows();
        pdgemm_(transposed, normal, min_mn, min_mn, m_global, 
        alpha, 
        matrix->local_data(), ia, ja, matrix->descriptor(),
        matrix->local_data(), ia, ja, matrix->descriptor(),
        beta,
        identity->local_data(), ia, ja, identity->descriptor());
    }
    else
    {
        auto n_global = matrix -> global_cols();
        pdgemm_(normal, transposed, min_mn, min_mn, n_global, 
        alpha, 
        matrix->local_data(), ia, ja, matrix->descriptor(),
        matrix->local_data(), ia, ja, matrix->descriptor(),
        beta,
        identity->local_data(), ia, ja, identity->descriptor());
    }

    return 1.0/min_mn * one_norm(identity);    
}

//
// Compute ||A - U * S * V'||
//
static double reconstruct_svd(std::shared_ptr<block_cyclic_mat_t> a, std::shared_ptr<block_cyclic_mat_t> u, std::shared_ptr<block_cyclic_mat_t> vt, const std::vector<double>& s)
{
    auto m      = a -> global_rows();
    auto n      = a -> global_cols();
    auto min_mn = std::min(m, n);
    auto s_dist = block_cyclic_mat_t::constant(a->grid(), min_mn, min_mn);
    
    for (blas_idx_t i = 1; i <= min_mn; i ++)
    {
        auto sigma = s[i-1];
        pdelset_(s_dist->local_data(), i, i, s_dist->descriptor(), sigma);
    }
    
    auto s_vt = block_cyclic_mat_t::constant(a->grid(), min_mn, n);

    // Compute Sigma * V^{Transpose}
    auto normal = 'N';
    auto ia     = static_cast<blas_idx_t>(1);
    auto ja     = static_cast<blas_idx_t>(1);    
    auto alpha  = 1.0;
    auto beta   = 0.0;
    pdgemm_(normal, normal, 
        min_mn, n, min_mn, 
        alpha, 
        s_dist -> local_data(), ia, ja, s_dist -> descriptor(), 
        vt -> local_data(), ia, ja, vt -> descriptor(),
        beta,
        s_vt -> local_data(), ia, ja, s_vt -> descriptor());
        
    // Compute U * Sigma * V^{Transposed} - A
    beta = -1.0;
    pdgemm_(normal, normal,
        m, n, min_mn,
        alpha,
        u -> local_data(), ia, ja, u -> descriptor(),
        s_vt -> local_data(), ia, ja, s_vt -> descriptor(),
        beta,
        a -> local_data(), ia, ja, a -> descriptor());

    return 1.0/min_mn * one_norm(a); 
}

static void svd_driver(blas_idx_t m_global, blas_idx_t n_global)
{
    auto grid = std::make_shared<blacs_grid_t>();    
    auto a    = block_cyclic_mat_t::random(grid, m_global, n_global);
    auto ia   = static_cast<blas_idx_t>(1);
    auto ja   = static_cast<blas_idx_t>(1);

    // Create a MxN matrix to hold A'
    auto acopy = block_cyclic_mat_t::constant(grid, m_global, n_global);

    // Copy A to A' since it will be overwritten during factorization
    std::copy_n(a->local_data(), a->local_size(), acopy->local_data());

    // Allocate U and V^{Transpose}
    // PDGESVD can currently only compute the 'economy' decomposition
    // of A, even though ScaLAPACK contains enough primitives to build the 'full' decomposition.
    // In an 'economy' decomposition, let MIN_MN = min(M, N), then:
    // U is a matrix of size M x MIN_MN
    // S is a diagonal square matrix of size MIN_MN
    // V^{Transpose} is a matrix of size MIN_MN x N

    auto min_mn = std::min(m_global, n_global);

    std::vector<double> sigma(min_mn);
    auto job_u = 'V';
    auto u     = block_cyclic_mat_t::constant(grid, m_global, min_mn);
    auto job_v = 'V';
    auto vt    = block_cyclic_mat_t::constant(grid, min_mn, n_global);
           
    blas_idx_t          lwork = 10;
    std::vector<double> work(lwork); 
    blas_idx_t          info  = 0;
    
    // Compute the workspace required
    lwork = -1;
    pdgesvd_(job_u, job_v, m_global, n_global, 
        a->local_data(), ia, ja, a -> descriptor(),
        sigma.data(),
        u->local_data(), ia, ja, u -> descriptor(),
        vt->local_data(), ia, ja, vt->descriptor(),
        work.data(),
        lwork, info);
    
    assert(info == 0);
    
    // Reallocate the workspace based on the returned estimate
    lwork = static_cast<blas_idx_t>(work[0]);
    work.resize(lwork);

    // Compute the SVD
    MPI_Barrier (MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    
    pdgesvd_(job_u, job_v, m_global, n_global, 
        a->local_data(), ia, ja, a -> descriptor(),
        sigma.data(),
        u->local_data(), ia, ja, u -> descriptor(),
        vt->local_data(), ia, ja, vt->descriptor(),
        work.data(),
        lwork, info);
    assert(info == 0);

    double t1 = MPI_Wtime() - t0;
  
    double t_glob;
    MPI_Reduce(&t1, &t_glob, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Verify SVD
    
    // Check if U'*U = I
    auto err_u = compute_residual(u, true);

    // Check if V'*V = I
    auto err_vt = compute_residual(vt, false);
       
    // Check if A = U * S * V'
    auto err_a = reconstruct_svd(acopy, u, vt, sigma);
    
    if (grid->iam() == 0) 
    {
        double gflops = pdgesvd_flops(m_global, n_global)/t_glob/grid->nprocs();
        printf("\n"
            "MATRIX SVD FACTORIZATION BENCHMARK SUMMARY\n"
            "==========================================\n"
            "M = %d\tN = %d\tNP = %d\tNP_ROW = %d\tNP_COL = %d\n"
            "Time for PxGESVD = %10.7f seconds\tGflops/Proc = %10.7f\n"
            "Error in U = %10.7f\tError in VT = %10.7f\tError in SVD = %10.7f\n",
            m_global, n_global, grid->nprocs(), grid->nprows(), grid->npcols(), 
            t_glob, gflops,
            err_u, err_vt, err_a);fflush(stdout);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    blas_idx_t m_global = 4096;
    blas_idx_t n_global = 4096;    

    if (argc > 1)
    {
        m_global = blas_idx_t(atol(argv[1]));
    }

    if (argc > 2)
    {
        n_global = blas_idx_t(atol(argv[2]));
    }

    svd_driver(m_global, n_global);
    MPI_Finalize();
}
