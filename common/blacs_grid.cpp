#include <cmath>
#include <cassert>

#include "blacs.h"
#include "blacs_grid.h"


blacs_grid_t::blacs_grid_t()
{    
	//两个参数都为输出，将当前进程号传递给m_iam，将进程总数传给m_nprocs
    blacs_pinfo_ (m_iam, m_nprocs);	

	//negone为输入，与一个特别的上下文向关联，如果找不到则忽略
	//zero为输入， 输入0表示处理默认的程序上下文(context)
	//m_ictxt为输出，The value the BLACS internal presently is. 
    blas_idx_t negone = -1, zero = 0, one = 1;    
    blacs_get_ (negone, zero, m_ictxt);
    
	//对总的进程数开放，使行进程数与列进程数的积为总进程数，且差不大
    m_nprows = blas_idx_t(sqrt(double(m_nprocs)));
	while(m_nprocs % m_nprows)
        m_nprows --;
	m_npcols = m_nprocs/m_nprows;
    assert(m_nprows * m_npcols == m_nprocs);

    const char* row_major = "Row";    
	// m_ictxt输入进程上下文id，如果不存在对应的进程上下文则创建
	// row_major 以行为主的顺序
	// m_nprows,m_npcols 行列进程数
    blacs_gridinit_ (m_ictxt, row_major, m_nprows, m_npcols);    

	// m_ictxt 为输入，输入进程上下文id
	// 通过参数传出 现在进程网格的行列数 与调用进程的行列坐标
    blacs_gridinfo_ (m_ictxt, m_nprows, m_npcols, m_myrow, m_mycol);
}

blas_idx_t blacs_grid_t::local_rows(blas_idx_t global_rows, blas_idx_t row_block_size, blas_idx_t row_offset /*= 0*/)
{
	//	计算并返回矩阵分块后在当前进程中的行坐标
	//	输入总行数，每个块的行数，当前进程格网中的行坐标，行偏移（一般为0），进程网格的总行数
	return numroc_ (global_rows, row_block_size, m_myrow, row_offset, m_nprows);    
}

blas_idx_t blacs_grid_t::local_cols(blas_idx_t global_cols, blas_idx_t col_block_size, blas_idx_t col_offset /*= 0*/)
{
	//	同上
    return numroc_ (global_cols, col_block_size, m_mycol, col_offset, m_npcols); 
}

blas_idx_t blacs_grid_t::nprows() const
{
    return m_nprows;
}

blas_idx_t blacs_grid_t::npcols() const
{
    return m_npcols;
}

blas_idx_t blacs_grid_t::myprow() const
{
    return m_myrow;
}

blas_idx_t blacs_grid_t::mypcol() const
{
    return m_mycol;
}

blacs_grid_t::~blacs_grid_t()
{
    blacs_gridexit_(m_ictxt);
}

blas_idx_t blacs_grid_t::iam() const
{
    return m_iam;
}

blas_idx_t blacs_grid_t::nprocs() const
{
    return m_nprocs;
}

blas_idx_t blacs_grid_t::context() const
{
    return m_ictxt;
}
