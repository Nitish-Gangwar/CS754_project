%%% This function contains the main function which calls slle inside it

% k_neighbours denotes the number of neigbours to be used in KNN
k_neighbours=10;

noise_R = 25.0;
% reduced_dim denotes the reduced number of dimension after
% dimensionality reduction
reduced_dim=2;

% theta taken uniformly at these angles projections will be taken
theta = linspace(0,180,180);

% https://in.mathworks.com/help/images/ref/phantom.html
% image = phantom('Modified Shepp-Logan',200);


% reading an image
% converting it to grayscale
% Lastly, normalizing it
image = mat2gray(rgb2gray(imread('/home/nitish/Desktop/CS754_advance_IP/project/203050037_203050069_Project/images/image1.png'))); % 

%%% Taking Radon projection at the angles theta.
R = radon(image, theta);
signal_std = std(R,0,'all');
noise_std = noise_R;

SNR = 10*log10(signal_std/noise_std);

R_noise = R + noise_std*randn(size(R))/5;
% Applying fourier slice theorem.
% http://www.cs.uoi.gr/~cnikou/Courses/Digital_Image_Processing/Chapter_05c_Image_Restoration_(Reconstruction_from_Projections).pdf
FTvecs = zeros(size(R_noise));
for i=1:length(R_noise)
    FTvecs(i,:) = fft(R_noise(i,:));
end


%%% Calling slle function for doing the core job.
[Y,Z] = slle(FTvecs, k_neighbours, reduced_dim);


angles_slle = sort(atand(Z(:,1)./Z(:,2)));
angles_slle_len = length(angles_slle);
final_theta = linspace(angles_slle(1),angles_slle(angles_slle_len),angles_slle_len);

% figure;
final_img = iradon(R_noise, final_theta);

subplot(1,2,1);
imshow(iradon(R_noise, theta));
title('Original Image');
subplot(1,2,2);
imshow(final_img);
title('Reconstructed by sLLE');

