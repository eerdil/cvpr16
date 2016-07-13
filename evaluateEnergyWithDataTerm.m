function minusLogpOfData = evaluateEnergyWithDataTerm(testImage, Psi, mask)
testImage = testImage / max(max(testImage));
binaryCurve = Psi < 0;
binaryCurveIn = binaryCurve .* (1 - mask);
temp1 = testImage(find(binaryCurveIn == 1));
c1 = mean(temp1);
binaryCurveOut = binaryCurveIn + mask;
temp2 = testImage(find(binaryCurveOut == 0));
c2 = mean(temp2);

term1 = sum(sum((temp1 - c1).^2));%/length(temp1);
term2 = sum(sum((temp2 - c2).^2));%/length(temp2);

minusLogpOfData = term1 + term2;
end