function [t, p, R2] = nipalspca(x,A)

xs = (x - mean(x))./std(x); % mean center and scale data
x = xs;
x0 = x;

k_s = size(x,2);
n_s = size(x,1);

tol = 1e-6;

R2 = zeros(1, A);
p = zeros(k_s,A);
t = zeros(n_s,A);

for a =1:A
    t(:,a) = x(:,1); %initialize t with any row from x
    
    delta_t = inf(size(t,1),1);
    
    while(any(delta_t>tol))
        
        p(:,a) = (t(:,a)'*x)/(t(:,a)'*t(:,a)); % calculate loadings
        
        p(:,a) = p(:,a)./sqrt(p(:,a)'*p(:,a)); % normalize loadings
        
        tnew = (x*p(:,a))/(p(:,a)'*p(:,a)); % calculate new scores
        
        delta_t = tnew - t(:,a); % compare the diff between old and new scores
        t(:,a) = tnew; % update scores
    end
    

    x = x-t(:,a)*p(:,a)'; % calculate residual
    
    res_x(:,:,a) = x;
    
    R2(a) = 1 - nansum(nansum(x.*x))./nansum(nansum(x0.*x0)); % Find cummulative R2 for each component
end



end