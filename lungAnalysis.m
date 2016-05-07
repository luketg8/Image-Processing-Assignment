function lungAnalysis(I)
image = imread(I); %image is read in
warning off; %warning messages are disabled
%% Task 1
convertedImage = 0.1*image(:,:,1)+0.45*image(:,:,2)+0.45*image(:,:,3); %convert image to gray scale, by reducing the values of rgb
%Check image, if image 4 apply Gaussian filter, else apply median filter to remove Salt and Pepper noise
if image == imread('4.jpg')
    convertedImage = double(convertedImage); %convert the image to a double
    sigma = 2; %set the value of sigma (standard deviation) to 2
    sz = 1; %set the size of the window to 1
    [x1,y1] = meshgrid(-sz:sz,-sz:sz); 
    M = size(x1,1)-1;
    N = size(y1,1)-1;
    exponent = -(x1.^2+y1.^2)/(2*sigma*sigma); %formulate the exponent of e
    exponent2 = exp(exponent)/(2*pi*sigma*sigma); %formulate the equation
    Output = zeros(size(convertedImage)); %initialise with 0s
    convertedImage = padarray(convertedImage,[sz sz]); %pad the image
    for i = 1:size(convertedImage, 1)-M;
        for j = 1:size(convertedImage, 2)-N;
            Temp = convertedImage(i:i+M,j:j+M)*exponent2; %apply the filter
            Output(i,j)=sum(Temp(:)); %output new values
        end
    end
    %(angeljohnsy.blogspot.com, 2015)
    convertedImage = uint8(Output);
    figure, imshow(convertedImage), title('Task 1, Noise Removed');
else
    [m, n] = size(convertedImage);
    medianWindow = zeros(3,3); %Create matrix of size 3x3, as this is default size
    %create function for median filtering
    for y = 2:n-1
        for x = 2:m-1
            medianWindow(1:3, 1:3) = convertedImage(x-1:x+1, y-1:y+1); %A segment of the image is covered by the window
            organise = reshape(medianWindow,[9,1]); %put the values in a row of 9
            sorted = sort(organise); %organise the values into order
            convertedImage(x,y) = sorted(5); %choose the median value to use
        end
    end
    figure, imshow(convertedImage), title('Task 1, Noise Removed');
end

%% Task 2
convertedImage2 = im2double(convertedImage); %convert the image data to double
threshold = graythresh(convertedImage);
convBw = im2bw(convertedImage2, threshold); %generate the threshold for the grayscale
seg = strel('disk', 12); %segment the area using a diamond shape with a 12 pixel radius
IO = imopen(convBw, seg); %apply the structured element to the image
bw = imfill(IO, [90 90]); %flood fill the image
reverse = ~bw; %reverse colours in image
figure, imshow(reverse), title('Segmented')

%% Task 3
convertedImage3 = im2bw(convertedImage, threshold); %Convert image to black and white, using the detected threshold
reverse1 = ~convertedImage3; %reverse colours in image, to make the circles easier to detect
reverse1 = imresize(reverse1, 2); %double the size of the image, to make the circles easier to detect
reverse1 = imclearborder(reverse1); %remove excess noise from the border of the image
[centres, radii] = imfindcircles(reverse1,[4 60], 'ObjectPolarity','dark','Sensitivity',0.74); %detect dark circles, with a sensitivity of 0.72
display('Task 3 Number of Circles');
figure, imshow(reverse1), title('Task 3 Highlight Circles');
display(numel(centres(:,1))) %count the number of elements in the array of 'centers', then divide by 2, to get the total number of circles
hold on
plot(centres(:,1), centres(:,2), 's', 'lineWidth', 1, 'Markersize', 8); %for every 'center' in the image, plot a bounding box, with a width of 2 and a size of 8
hold off

%% Task 4
reverse2 = ~reverse1; %reverse the values in the image
CC = bwconncomp(reverse2); %identify the connected components in the image
S = regionprops(CC, 'Area'); %get the area of each connected component
L = labelmatrix(CC); %create a label matrix for each element
reverse2 = ismember(L, find([S.Area] < 1500)); %remove every element that has an area greater than 1500 pixels
figure, imshow(reverse1), title('Task 4 Circles Highlighted');
viscircles(centres, radii,'Color','g'); %Visualise the circles onto the last output figure, using the colour green
figure, imshow(reverse2), title('Task 4 Circles Highlighted and Circles Segmented');
viscircles(centres, radii,'Color','g'); %Visualise the circles onto the last output figure, using the colour green
end