function narrowBand = createNarrowBand(Psi, bandThickness)
se = strel('square', bandThickness);
PsiDilated = imdilate(Psi > 0, se);
PsiEroded = imerode(Psi > 0, se);

narrowBand = PsiDilated - PsiEroded;
% narrowBand
end