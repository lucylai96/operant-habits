function [ emp_entropy ] = calc_entropy(samples)
%CALC_ENTROPY 
    [ w, e ] = histcounts( samples );

    bin_size    = e(2)-e(1);
    w = w ./ sum(w);
    emp_entropy = 0.00;
    for i = 1:length(w)
        if( w(i) > 0.0 )
            emp_entropy = emp_entropy + (w(i) * log(w(i)/bin_size));
        end
    end
    emp_entropy = -1 * emp_entropy;
end