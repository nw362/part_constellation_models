function [ x,y ] = part_location_from_gmap(gmap)
% part_location_from_gmap Returns part location from gradient map 

  if any(gmap(:)~=0)
    % Apply gaussian filter with hsize = [5 5] and sigma = 2
    G = fspecial('gaussian',[20 20],3);
    gmap = imfilter(gmap,G,'same');

    [est_y, est_x] =find(max(gmap(:))==gmap,1,'last');
    y = int32(est_y(1));
    x = int32(est_x(1));
  else
    y = -1;
    x = -1;
  end
end
