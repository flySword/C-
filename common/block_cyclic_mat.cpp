#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "block_cyclic_mat.h"
#include "scalapack.h"

block_cyclic_mat_t::block_cyclic_mat_t(std::shared_ptr<blacs_grid_t> grid, blas_idx_t global_rows, blas_idx_t global_cols, blas_idx_t mb, blas_idx_t nb, fill_t fill /*= EMPTY*/, double alpha /*= 0.0*/) : m_grid(grid), m_global_rows(global_rows), m_global_cols(global_cols)
{
	m_mb = mb;	// 行的块数  都等于64  TODO
	m_nb = nb;	// 列的块数
	m_local_rows = m_grid->local_rows(m_global_rows, mb);	//矩阵分块后在当前进程中的行坐标
	m_local_cols = m_grid->local_cols(m_global_cols, nb);

	m_desc[DTYPE_] = 1;		//Descriptor type ( =1 for dense matrices).
	m_desc[CTXT_] = m_grid->context();	//BLACS context handle for the process grid. 
	m_desc[M_] = global_rows;	//Number of rows in the global matrix A.
	m_desc[N_] = global_cols;	//Number of columns in the global matrix A.
	m_desc[MB_] = mb;	//Row blocking factor. 矩阵分解后的行块数  64 TODO
	m_desc[NB_] = nb;	//Column blocking factor. 矩阵分解后的列块数  64 TODO
	m_desc[RSRC_] = 0;		//Process row over which the first row of the global matrix A is distributed. 
	m_desc[CSRC_] = 0;		//Process column over which the first column of the global matrix A is distributed. 
	m_desc[LLD_] = m_local_rows;	//Leading dimension of the local matrix A. 

	m_local_size = m_local_rows * m_local_cols;
	m_local_data.resize(m_local_size);

	switch (fill)
	{
	case CONSTANT:
	{
		std::fill(m_local_data.begin(), m_local_data.end(), alpha);
		break;
	}
	case DIAGONAL:
	{
		char uplo = 'A';
		blas_idx_t ia = 1, ja = 1;
		double zero = 0.0;
		pdlaset_(uplo, m_global_rows, m_global_cols, zero, alpha, m_local_data.data(), ia, ja, m_desc);
		break;
	}
	case RANDOM:
	{
		std::mt19937_64 engine(1000 * m_grid->iam());
		std::uniform_real_distribution<double> rng;
		std::generate(m_local_data.begin(), m_local_data.end(), [&]() {return rng(engine); });
		break;
	}

	}
}

block_cyclic_mat_t::block_cyclic_mat_t(std::shared_ptr<blacs_grid_t> grid, blas_idx_t global_rows, blas_idx_t global_cols, double *array, blas_idx_t mb, blas_idx_t nb, fill_t fill /*= EMPTY*/, double alpha /*= 0.0*/) : m_grid(grid), m_global_rows(global_rows), m_global_cols(global_cols)
{
	m_mb = mb;	// 行的块数  都等于64  TODO
	m_nb = nb;	// 列的块数
	m_local_rows = m_grid->local_rows(m_global_rows, mb);	//矩阵分块后在当前进程中的行坐标
	m_local_cols = m_grid->local_cols(m_global_cols, nb);

	m_desc[DTYPE_] = 1;		//Descriptor type ( =1 for dense matrices).
	m_desc[CTXT_] = m_grid->context();	//BLACS context handle for the process grid. 
	m_desc[M_] = global_rows;	//Number of rows in the global matrix A.
	m_desc[N_] = global_cols;	//Number of columns in the global matrix A.
	m_desc[MB_] = mb;	//Row blocking factor. 矩阵分解后的行块数  64 TODO
	m_desc[NB_] = nb;	//Column blocking factor. 矩阵分解后的列块数  64 TODO
	m_desc[RSRC_] = 0;		//Process row over which the first row of the global matrix A is distributed. 
	m_desc[CSRC_] = 0;		//Process column over which the first column of the global matrix A is distributed. 
	m_desc[LLD_] = m_local_rows;	//Leading dimension of the local matrix A. 

	m_local_size = m_local_rows * m_local_cols;
	m_local_data.resize(m_local_size);

	switch (fill)
	{
	case CONSTANT:
	{
		std::fill(m_local_data.begin(), m_local_data.end(), alpha);
		break;
	}
	case DIAGONAL:
	{
		char uplo = 'A';
		blas_idx_t ia = 1, ja = 1;
		double zero = 0.0;
		pdlaset_(uplo, m_global_rows, m_global_cols, zero, alpha, m_local_data.data(), ia, ja, m_desc);
		break;
	}
	case RANDOM:
	{
		std::mt19937_64 engine(1000 * m_grid->iam());
		std::uniform_real_distribution<double> rng;
		std::generate(m_local_data.begin(), m_local_data.end(), [&]() {return rng(engine); });
		break;
	}
	case ARRAY:
	{
		double *a = array;
		//int index = 0;
		for (std::vector<double>::iterator i = m_local_data.begin(); i != m_local_data.end(); i++,a++)
		{
			*i = *a;
		}
	}

	}
}

void block_cyclic_mat_t::print() const
{
	for (int i = 0; i < m_local_size; i++)
	{
		printf("local[%d] = %lf\n", i, m_local_data[i]); fflush(stdout);
	}
}

blas_idx_t block_cyclic_mat_t::local_size() const
{
	return m_local_size;
}

blas_idx_t block_cyclic_mat_t::local_rows() const
{
	return m_local_rows;
}

blas_idx_t block_cyclic_mat_t::local_cols() const
{
	return m_local_cols;
}

blas_idx_t block_cyclic_mat_t::global_rows() const
{
	return m_global_rows;
}

blas_idx_t block_cyclic_mat_t::global_cols() const
{
	return m_global_cols;
}

blas_idx_t block_cyclic_mat_t::row_block_size() const
{
	return m_mb;
}

blas_idx_t block_cyclic_mat_t::col_block_size() const
{
	return m_nb;
}

double* block_cyclic_mat_t::local_data()
{
	return m_local_data.data();
}

blas_idx_t* block_cyclic_mat_t::descriptor()
{
	return m_desc;
}

std::shared_ptr<blacs_grid_t> block_cyclic_mat_t::grid()
{
	return m_grid;
}

std::shared_ptr<block_cyclic_mat_t> block_cyclic_mat_t::random(std::shared_ptr<blacs_grid_t> grid, blas_idx_t global_rows, blas_idx_t global_cols)
{
	return std::shared_ptr<block_cyclic_mat_t>(new block_cyclic_mat_t(grid, global_rows, global_cols, s_block_size, s_block_size, RANDOM, 0.0));
}

std::shared_ptr<block_cyclic_mat_t>  block_cyclic_mat_t::createWithArray(std::shared_ptr<blacs_grid_t> grid, blas_idx_t global_rows, blas_idx_t global_cols, double *array)
{
	return std::shared_ptr<block_cyclic_mat_t>(new block_cyclic_mat_t(grid, global_rows, global_cols, array, s_block_size, s_block_size, ARRAY, 0.0));
}

std::shared_ptr<block_cyclic_mat_t> block_cyclic_mat_t::constant(std::shared_ptr<blacs_grid_t> grid, blas_idx_t global_rows, blas_idx_t global_cols, double alpha /* = 0.0 */)
{
	return std::shared_ptr<block_cyclic_mat_t>(new block_cyclic_mat_t(grid, global_rows, global_cols, s_block_size, s_block_size, CONSTANT, alpha));
}

std::shared_ptr<block_cyclic_mat_t> block_cyclic_mat_t::diagonal(std::shared_ptr<blacs_grid_t> grid, blas_idx_t global_rows, blas_idx_t global_cols, double alpha /*= 1.0*/)
{
	return std::shared_ptr<block_cyclic_mat_t>(new block_cyclic_mat_t(grid, global_rows, global_cols, s_block_size, s_block_size, DIAGONAL, alpha));
}
