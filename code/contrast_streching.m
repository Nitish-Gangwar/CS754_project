function output_image = contrast_streching(image)
    minimum = min(image, [], 'all');
    maximum = max(image, [], 'all');
    output_image = (image - minimum)/(maximum - minimum);
end