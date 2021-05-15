k_neighbours=10;

reduced_dim=2;

%theta=180.0*rand(no_of_angles,1);
% theta = randi([0 179],1,35);
theta = 0:1:179;

% https://in.mathworks.com/help/images/ref/phantom.html
% Uncomment the required line to read the image
% image = phantom('Modified Shepp-Logan',200); % Shepp-Logan Phantom image
% image = mat2gray(rgb2gray(imread('../images/image1.png'))); % Brain MR
image = mat2gray(imread('/home/nitish/Desktop/CS754_advance_IP/project/203050037_203050069_Project/images/surf_z_1.png')); % ribosome cryoEM
% image = mat2gray(imread('../images/proj0.png')); % Cyanophage cryoEM

R = radon(image, theta);

% Applying fourier slice theorem.
% http://www.cs.uoi.gr/~cnikou/Courses/Digital_Image_Processing/Chapter_05c_Image_Restoration_(Reconstruction_from_Projections).pdf
FTvecs = zeros(size(R));
for i=1:length(R)
    FTvecs(i,:) = fft(R(i,:));
end

Z = lle(FTvecs, k_neighbours, reduced_dim);
% X = R;
% K = k_neighbours;
% d = reduced_dim;
Z = real(Z);
angles_lle = sort(atand(Z(2,:)./Z(1,:)));

final_img = iradon(R, angles_lle);
% imshow(mat2gray(img));

subplot(1,2,1);
imshow(image);
title('Original Image');
subplot(1,2,2);
imshow(final_img);
title('Reconstructed by LLE');

