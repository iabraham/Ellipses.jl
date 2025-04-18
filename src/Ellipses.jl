module Ellipses

using LinearAlgebra, JuMP, Ipopt

# Define an ellipse in quadratic form 
struct EllipseQform{T}

	# Three parameter form (ellipse always centered at origin)
	params::Vector{T}

	# Canonical form (major/minor axis length + rotation)
	canon::Vector{T}

	# Function to evaluate form 
	func::Function
end

"""Constructor for `EllipseQform` struct"""
function EllipseQform(A::T, B::T, C::T) where T
	theta = atan(B/(A-C))/2
	rr = B / sin(2*theta)
	asqr, bsqr  = 1/((A+C+rr)/2.0), 1/((A+C-rr)/2.0)
	a = sqrt(asqr < 0 ? 0 : asqr)
	b = sqrt(bsqr < 0 ? 0 : bsqr)
    if a == 0 && b==0
        A, B, C = 1, 0, 1
    end
	fun(x, y) = A*x^2+ B*x*y + C*y^2
	EllipseQform{T}([A,B,C], [a, b, theta], fun)
end	
export EllipseQform

# Define an ellipse in parametric form 
struct EllipsePform{T}
    semiaxis_lengths::Vector{T}
    center::Vector{T}
    ccw_angle::T
end

# Solve nonlinear optimization problem to fit ellipse
"""Function fits an ellipse to vector `v`"""
function fit_ellipse(v::Vector{T}) where T
	xy = map(z-> (real(z), imag(z)), v)
	ellipse = Model(Ipopt.Optimizer)
	@variable(ellipse, a)
	@variable(ellipse, b)
	@variable(ellipse, c)
	@NLobjective(ellipse, Min, sum((a*x^2+2*b*x*y+c*y^2-1)^2 for (x,y) in xy))
	@NLconstraint(ellipse, a*c-b^2>=0)
	set_silent(ellipse)
	optimize!(ellipse)
	return EllipseQform(map(value, (a, b, c))...)
end
export fit_ellipse

# Helper function to convert between parametric & quadratic forms 
"""Function returns the elementwise pseudoinverse of vector `v`"""
function elementwise_pseudoinvert(v::AbstractArray, tol=1e-10)     
    m = maximum(abs.(v));
    m == 0 && return v
    v = v ./ m;
    reciprocal = 1 ./ v;
    reciprocal[abs.(reciprocal) .>= 1/tol] .= 0;   

    return reciprocal / m;
end

# Function conversts a quadratic form ellipse to parametric form ellipse
# Parametric form is easier to draw/plot
# Quadratic form is easier for function evaluations
"""Function converts between qform and pform ellipse structs."""
function quad2parametric(qform::EllipseQform{T}) where T
    S = [qform.params[1] qform.params[2]/2
		 qform.params[2]/2 qform.params[3]]
    f = eigen(S)
    V = f.vectors
    D = f.values                
 
    semiaxis_lengths = sqrt.(abs.(elementwise_pseudoinvert(D)))
    p = sortperm(semiaxis_lengths, rev=true)
    sorted_semiaxes = semiaxis_lengths[p]
    sorted_eig_vecs = V[:,p]
    major_axis = sorted_eig_vecs[:,1]
    ccw_angle = atan(major_axis[2], major_axis[1])
 
    return EllipsePform{T}(sorted_semiaxes, vec([0,0]), ccw_angle)
end
export quad2parametric

"""Returns a 2D rotation matrix"""
function rotation_mat(angle::Real; ccw=true)
    ccw ? [cos(angle) -sin(angle);
           sin(angle) cos(angle)] : [cos(angle) sin(angle);
		                            -sin(angle) cos(angle)];
end

""""Returns x-y points to be able to plot an ellipse"""
function ellipse_to_plot_points(ellipse::EllipsePform; n=1000::Int)
    theta_plot_vals = range(0, 2*pi, length=n)
    unit_circle = [cos.(theta_plot_vals) sin.(theta_plot_vals)]'
    onaxis_ellipse = vec(ellipse.semiaxis_lengths) .* unit_circle
    U = rotation_mat(ellipse.ccw_angle)
    rotated_ellipse = (U * onaxis_ellipse)'
    shifted_ellipse = vec(ellipse.center)' .+ rotated_ellipse

    return shifted_ellipse
end
export ellipse_to_plot_points

end
