def eval_mcol(rho, m, xc_type, eval_col, deriv=1, spin_grid=None):
    '''
    Evaluates the multi-collinear functional and its derivatives based on
    the corresponding collinear functional 

    Parameters
    ----------
    rho : np.ndarray
        The density and its derivaties on grids. Rho has the shape (*, Ngrids).
        Depending on the type of XC functional, the leading dimension can be 1
        (LDA), 4 (GGA), or 6 (MGGA):
            [0] the density
            [1] grad_x - the gradient of density along x axis
            [2] grad_y - the gradient of density along y axis
            [3] grad_z - the gradient of density along z axis
            [4] lapl - the laplacian of the density
            [5] tau - the kinetic energy density

    m : np.ndarray
        The spin density and its derivaties on grids. m has the shape (*, 3, Ngrids).
        The value 3 in the second dimension corresponds to the 3 components of
        spin density. Depending on the type of XC functional, the leading
        dimension can be 1 (LDA), 4 (GGA), or 6 (MGGA):
            [0] the spin density
            [1] grad_x - the gradient of spin density along x axis
            [2] grad_y - the gradient of spin density along y axis
            [3] grad_z - the gradient of spin density along z axis
            [4] lapl - the laplacian of the spin density
            [5] tau - the kinetic energy spin density

    xc_type: string
        The type of XC functional: LDA, GGA, or MGGA.

    eval_col: function
        The function to evaluate the collinear functional.
    '''

    raise NotImplementedError
