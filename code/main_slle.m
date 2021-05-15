%%% This function contains the main function which calls slle inside it

% k_neighbours denotes the number of neigbours to be used in KNN
k_neighbours=10;


% reduced_dim denotes the reduced number of dimension after
% dimensionality reduction
reduced_dim=2;

% theta taken uniformly at these angles projections will be taken
theta = linspace(0,180,180);
% theta = 180*rand(180,1);

% https://in.mathworks.com/help/images/ref/phantom.html


% reading an image
% converting it to grayscale
% Lastly, normalizing it

% Uncomment the required line to read the image
image = phantom('Modified Shepp-Logan',200); % Shepp-Logan Phantom image
% image = mat2gray(rgb2gray(imread('../images/image1.png'))); % Brain MR
% image = mat2gray(imread('../images/surf_z_1.png')); % ribosome cryoEM
% image = mat2gray(imread('../images/proj0.png')); % Cyanophage cryoEM

% imshow(image);


%%% Taking Radon projection at the angles theta.
R = radon(image, theta);

% Applying fourier slice theorem.
% http://www.cs.uoi.gr/~cnikou/Courses/Digital_Image_Processing/Chapter_05c_Image_Restoration_(Reconstruction_from_Projections).pdf
FTvecs = zeros(size(R));
for i=1:length(R)
    FTvecs(i,:) = fft(R(i,:));
end

%%% Calling slle function for doing the core job.
[Y,Z] = slle(FTvecs, k_neighbours, reduced_dim);
% X = (FTvecs);
% K = k_neighbours;
% d = reduced_dim;



angles_slle = sort(atand(Z(:,1)./Z(:,2)));
angles_slle_len = length(angles_slle);
final_theta = linspace(angles_slle(1),angles_slle(angles_slle_len),angles_slle_len);

% figure;
final_img = iradon(R, final_theta);
% imshow((final_img));

% https://in.mathworks.com/help/images/ref/centercropwindow2d.html
corrected_img = iradon(R, final_theta+abs(angles_slle(1)));
% figure;
win1 = centerCropWindow2d(size(corrected_img),size(image));
corrected_crop_img = imcrop(corrected_img,win1);
% imshow(corrected_crop_img);

[one,two] = size(image);

MSE = sum((corrected_crop_img-image).^2,'all')/(one*two);

PSNR = 20*log10(max(image,[],'all')/sqrt(MSE));

fprintf('<-------------->\n')
fprintf('MSE: %f\n', MSE);
fprintf('PSNR: %f\n', PSNR);

subplot(1,2,1);
imshow(image);
title('Original Image');
subplot(1,2,2);
imshow(final_img);
title('Reconstructed by sLLE');

