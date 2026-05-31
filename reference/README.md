# DNS 參考解 provenance

來源：home-gpu:~/cfd_solver/lid_driven_cavity/output_regularized_r7
- Solver：Lethe (deal.II), method=steady, Newton tol 1e-8
- 網格：128×128 (refinement 7), Q1-Q1
- Re=1000 (nu=0.001, U=L=1)
- Lid 剖面：g(x)=1-cosh(10(x-0.5))/cosh(5)（與 PINN lid_r=10 相同）
- 抽取：vtk 讀 .00001 收斂解 → src/pinn_cavity/data/dns_re1000_r10.npz (16641 唯一點)

取代 Ghia 1982（sharp lid）以達 apples-to-apples 驗證。
