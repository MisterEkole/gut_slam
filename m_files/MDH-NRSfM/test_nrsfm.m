% Sample Input Data
M = 3; % Number of views
N = 5; % Number of tracked points

% Neighborhood matrix (replace with your actual data)
IDX = [1 2 3; 2 1 3; 3 1 2; 4 5 3; 5 4 3];

% Cell array of normalized point correspondences
m = cell(1, M);
for i = 1:M
    m{i} = rand(2, N); % Replace with actual data
end

% Cell array of point visibility
vis = cell(1, M);
for i = 1:M
    vis{i} = logical(randi([0 1], 1, N)); % Replace with actual data
end

% Solver choice (optional, defaults to 'mosek')
solver = 'mosek';

% Call NrSfM function
[mu, D] = NrSfM(IDX, m, vis, solver);

% Display Results
disp('Depth Matrix (mu):');
disp(mu);

disp('Maximum Distance Matrix (D):');
disp(D);
