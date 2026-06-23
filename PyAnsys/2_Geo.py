import math
import os

# Manually entered parametric study data
data = [
    {"Curvature_Radius": 376.770, "Aneurysm_width": 10.905, "neck": 1.970},
    {"Curvature_Radius": 626.085, "Aneurysm_width": 20.665, "neck": 2.425},
    {"Curvature_Radius": 566.735, "Aneurysm_width": 18.655, "neck": 3.405},
    {"Curvature_Radius": 324.590, "Aneurysm_width": 15.260, "neck": 3.245},
    {"Curvature_Radius": 223.250, "Aneurysm_width": 16.725, "neck": 2.640},
    {"Curvature_Radius": 603.665, "Aneurysm_width": 14.980, "neck": 3.425},
    {"Curvature_Radius": 524.595, "Aneurysm_width": 11.075, "neck": 2.490},
    {"Curvature_Radius": 768.470, "Aneurysm_width": 23.590, "neck": 3.520},
    {"Curvature_Radius": 615.490, "Aneurysm_width": 21.675, "neck": 2.600},
    {"Curvature_Radius": 398.675, "Aneurysm_width": 17.950, "neck": 2.160},
    {"Curvature_Radius": 671.235, "Aneurysm_width": 13.220, "neck": 0.670},
    {"Curvature_Radius": 218.640, "Aneurysm_width": 17.315, "neck": 1.910},
    {"Curvature_Radius": 110.450, "Aneurysm_width": 18.585, "neck": 3.320},
    {"Curvature_Radius": 640.625, "Aneurysm_width": 22.790, "neck": 3.710},
    {"Curvature_Radius": 568.700, "Aneurysm_width": 19.735, "neck": 3.780},
    {"Curvature_Radius": 169.235, "Aneurysm_width": 13.730, "neck": 0.920},
    {"Curvature_Radius": 297.310, "Aneurysm_width": 13.670, "neck": 1.840},
    {"Curvature_Radius": 632.935, "Aneurysm_width": 18.615, "neck": 2.705},
    {"Curvature_Radius": 661.945, "Aneurysm_width": 13.530, "neck": 3.770},
    {"Curvature_Radius": 691.095, "Aneurysm_width": 13.050, "neck": 1.970},
    {"Curvature_Radius": 527.970, "Aneurysm_width": 16.690, "neck": 1.570},
    {"Curvature_Radius": 461.910, "Aneurysm_width": 11.725, "neck": 1.115},
    {"Curvature_Radius": 314.850, "Aneurysm_width": 21.355, "neck": 1.215},
    {"Curvature_Radius": 575.250, "Aneurysm_width": 12.480, "neck": 1.915},
    {"Curvature_Radius": 401.565, "Aneurysm_width": 18.770, "neck": 1.395},
    {"Curvature_Radius": 219.585, "Aneurysm_width": 11.955, "neck": 2.295},
    {"Curvature_Radius": 138.695, "Aneurysm_width": 21.660, "neck": 2.765},
    {"Curvature_Radius": 197.345, "Aneurysm_width": 12.265, "neck": 3.365},
    {"Curvature_Radius": 718.995, "Aneurysm_width": 20.625, "neck": 2.490},
    {"Curvature_Radius": 306.200, "Aneurysm_width": 12.130, "neck": 0.670},
    {"Curvature_Radius": 508.770, "Aneurysm_width": 11.425, "neck": 3.465},
    {"Curvature_Radius": 188.695, "Aneurysm_width": 11.735, "neck": 3.625},
    {"Curvature_Radius": 351.745, "Aneurysm_width": 16.430, "neck": 3.765},
    {"Curvature_Radius": 362.135, "Aneurysm_width": 17.120, "neck": 1.375},
    {"Curvature_Radius": 446.530, "Aneurysm_width": 10.835, "neck": 1.720},
    {"Curvature_Radius": 278.470, "Aneurysm_width": 21.075, "neck": 3.755},
    {"Curvature_Radius": 326.050, "Aneurysm_width": 15.770, "neck": 2.870},
    {"Curvature_Radius": 702.410, "Aneurysm_width": 23.580, "neck": 0.725},
    {"Curvature_Radius": 731.185, "Aneurysm_width": 18.230, "neck": 0.935},
]

# Base save path for files
# save_path = "X:/Ansys/Geometry_Generation/"
# os.makedirs(save_path)

# Iterate through geometries and generate files
for i, geom in enumerate(data, start=1):
    # Assign parameters
    Artery_length = 70.0  # Use physical values
    Curvature_Radius = geom["Curvature_Radius"]
    Artery_Diameter = 5.0  # Physical artery diameter
    Aneurysm_width = geom["Aneurysm_width"]
    neck = geom["neck"]


    Center_distance=(-neck/2)+(Aneurysm_width/2)

    print(Center_distance)


    import math

    R = Curvature_Radius

    L = Artery_length


    # Central angle (theta) in radians

    theta = 2 * math.asin(L / (2 * R))


    # Arc length (S)

    ArcLength = R * theta


    # Divide the arc length by 2 for extrusion length

    ExtrusionLength = ArcLength / 2


    print(ExtrusionLength)


    # Set Sketch Plane

    sectionPlane = Plane.PlaneXY

    result = ViewHelper.SetSketchPlane(sectionPlane, Info1)

    # EndBlock



    # Sketch Line

    start = Point2D.Create(MM(-15), MM(10))

    end = Point2D.Create(MM(13), MM(10))

    result = SketchLine.Create(start, end)


    curveSelList = Curve1

    result = Constraint.CreateHorizontal(curveSelList)

    # EndBlock


    # 

    # Create Length Dimension

    dimTarget = Curve1

    alignment = DimensionAlignment.Aligned

    result = Dimension.CreateLength(dimTarget, alignment)

    # EndBlock


    # Edit dimension

    selDimension = SketchDimension1

    newValue = MM(Artery_length)

    result = Dimension.Modify(selDimension, newValue)

    # EndBlock


    # Symmetric Constraint

    baseSel = SelectionPoint.Create(DatumLine1)

    targetSel = Selection.Create(CurvePoint1, CurvePoint2)


    options = SymmetricConstraintOptions()

    options.SymmetricEnds = True

    result = Constraint.CreateSymmetric(baseSel, targetSel, options)

    # EndBlock



    # Sketch Circle

    origin = Point2D.Create(MM(-43), MM(-35))

    result = SketchCircle.Create(origin, MM(30.8706980808663))

    # EndBlock


    # 

    # Create Diameter Dimension

    dimTarget = Curve2

    result = Dimension.CreateDiameter(dimTarget)

    # EndBlock


    # Edit dimension

    selDimension = SketchDimension2

    newValue = MM(Curvature_Radius)

    result = Dimension.Modify(selDimension, newValue)

    # EndBlock


    # Coincident Constraint

    baseSel = SelectionPoint.Create(Curve2, 0.392699081698724)

    targetSel = SelectionPoint.Create(CurvePoint3)


    result = Constraint.CreateCoincident(baseSel, targetSel)

    # EndBlock


    # Coincident Constraint

    baseSel = SelectionPoint.Create(Curve2, 1.91142725806528)

    targetSel = SelectionPoint.Create(CurvePoint4)


    result = Constraint.CreateCoincident(baseSel, targetSel)

    # EndBlock



    # Set Sketch Plane

    sectionPlane = Plane.Create(Frame.Create(Point.Create(MM(0), MM(0), MM(0)), 

        Direction.DirY, 

        -Direction.DirZ))

    result = ViewHelper.SetSketchPlane(sectionPlane, Info2)

    # EndBlock


    # Sketch Circle

    origin = Point2D.Create(MM(20), MM(0))

    result = SketchCircle.Create(origin, MM(4))


    baseSel = SelectionPoint.Create(CurvePoint5)

    targetSel = SelectionPoint.Create(DatumLine2)


    result = Constraint.CreateCoincident(baseSel, targetSel)

    # EndBlock







    # Project to Sketch

    selection = Curve9

    plane = Plane.Create(Frame.Create(Point.Create(MM(0), MM(0), MM(0)), 

        Direction.DirY, 

        -Direction.DirZ))

    result = ProjectToSketch.Create(selection, plane)

    # EndBlock




    # Coincident Constraint

    baseSel = SelectionPoint.Create(CurvePoint11)

    targetSel = SelectionPoint.Create(CurvePoint12)


    result = Constraint.CreateCoincident(baseSel, targetSel)

    # EndBlock







    # 

    # Create Diameter Dimension

    dimTarget = Curve3

    result = Dimension.CreateDiameter(dimTarget)

    # EndBlock


    # Edit dimension

    selDimension = SketchDimension3

    newValue = MM(Artery_Diameter)

    result = Dimension.Modify(selDimension, newValue)

    # EndBlock



    # Solidify Sketch

    mode = InteractionMode.Solid

    result = ViewHelper.SetViewMode(mode, Info3)

    # EndBlock




    # Sweep 1 Face

    selection = Face2

    trajectories = Curve4

    options = SweepCommandOptions()

    options.ExtrudeType = ExtrudeType.Add

    options.Select = True

    result = Sweep.Execute(selection, trajectories, MM(ExtrusionLength), options, Info7)

    # EndBlock





    # Mirror

    selection = Body5

    mirrorPlane = Face3

    options = MirrorOptions()

    result = Mirror.Execute(selection, mirrorPlane, options, Info8)

    # EndBlock





    # Set Sketch Plane

    sectionPlane = Plane.PlaneXY

    result = ViewHelper.SetSketchPlane(sectionPlane, Info4)

    # EndBlock






    # Sketch Line

    start = Point2D.Create(MM(-5.96995597488409E-15), MM(42.5))

    end = Point2D.Create(MM(-7.11853265136092E-15), MM(45.5))

    result = SketchLine.Create(start, end)


    baseSel = SelectionPoint.Create(CurvePoint6)

    targetSel = SelectionPoint.Create(DatumLine3)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    baseSel = SelectionPoint.Create(CurvePoint7)

    targetSel = SelectionPoint.Create(DatumLine3)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    curveSelList = Curve5

    result = Constraint.CreateVertical(curveSelList)


    baseSel = SelectionPoint.Create(CurvePoint6)

    targetSel = SelectionPoint.Create(Curve6, 4.71238898038469)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    baseSel = SelectionPoint.Create(Curve5)

    targetSel = SelectionPoint.Create(Curve6)


    result = Constraint.CreatePerpendicular(baseSel, targetSel)

    # EndBlock


    # 

    # Create Length Dimension

    dimTarget = Curve5

    alignment = DimensionAlignment.Aligned

    result = Dimension.CreateLength(dimTarget, alignment)

    # EndBlock


    # Edit dimension

    selDimension = SketchDimension4

    newValue = MM(Center_distance)

    result = Dimension.Modify(selDimension, newValue)

    # EndBlock



    # Sketch Circle

    origin = Point2D.Create(MM(0), MM(34))

    result = SketchCircle.Create(origin, MM(13.6014705087354))


    baseSel = SelectionPoint.Create(CurvePoint24)

    targetSel = SelectionPoint.Create(DatumLine8)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    baseSel = SelectionPoint.Create(CurvePoint24)

    targetSel = SelectionPoint.Create(CurvePoint27)


    result = Constraint.CreateCoincident(baseSel, targetSel)

    # EndBlock


    # 

    # Create Diameter Dimension

    dimTarget = Curve25

    result = Dimension.CreateDiameter(dimTarget)

    # EndBlock


    # Edit dimension

    selDimension = SketchDimension6

    newValue = MM(Aneurysm_width)

    result = Dimension.Modify(selDimension, newValue)

    # EndBlock




    # Sketch Line

    start = Point2D.Create(MM(7.65378971138986E-16), MM(46.5))

    end = Point2D.Create(MM(-2.29613691341696E-15), MM(21.5))

    result = SketchLine.Create(start, end)


    baseSel = SelectionPoint.Create(CurvePoint28)

    targetSel = SelectionPoint.Create(Curve25, 1.5707963267949)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    baseSel = SelectionPoint.Create(CurvePoint29)

    targetSel = SelectionPoint.Create(Curve25, 4.71238898038469)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    baseSel = SelectionPoint.Create(Curve26)

    targetSel = SelectionPoint.Create(Curve25)


    result = Constraint.CreatePerpendicular(baseSel, targetSel)


    baseSel = SelectionPoint.Create(Curve26)

    targetSel = SelectionPoint.Create(Curve27)


    result = Constraint.CreatePerpendicular(baseSel, targetSel)


    baseSel = SelectionPoint.Create(CurvePoint28)

    targetSel = SelectionPoint.Create(DatumLine8)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    baseSel = SelectionPoint.Create(CurvePoint29)

    targetSel = SelectionPoint.Create(DatumLine8)


    result = Constraint.CreateCoincident(baseSel, targetSel)


    curveSelList = Curve26

    result = Constraint.CreateVertical(curveSelList)

    # EndBlock


    # Set Sketch Curve Construction

    selection = Curve28

    result = SketchHelper.SetConstructionCurve(selection, True)

    # EndBlock


    # Solidify Sketch

    mode = InteractionMode.Solid

    result = ViewHelper.SetViewMode(mode, Info11)

    # EndBlock


    # Revolve 1 Face

    selection = Face8

    axisSelection = Edge5

    axis = RevolveFaces.GetAxisFromSelection(selection, axisSelection)

    options = RevolveFaceOptions()

    options.ExtrudeType = ExtrudeType.Add

    result = RevolveFaces.Execute(selection, axis, DEG(360), options)

    # EndBlock


    # Delete Objects

    selection = CurveFolder1

    result = Delete.Execute(selection)

    # EndBlock


    # Delete Objects

    selection = Body10

    result = Delete.Execute(selection)

    # EndBlock



    # Create Named Selection Group

    primarySelection = Face9

    secondarySelection = Selection.Empty()

    result = NamedSelection.Create(primarySelection, secondarySelection, "Group1")

    # EndBlock


    # Rename Named Selection

    result = NamedSelection.Rename("Group1", "blood-inlet")

    # EndBlock


    # Create Named Selection Group

    primarySelection = Face10

    secondarySelection = Selection.Empty()

    result = NamedSelection.Create(primarySelection, secondarySelection, "Group1")

    # EndBlock


    # Rename Named Selection

    result = NamedSelection.Rename("Group1", "blood-outlet")

    # EndBlock

    # Save File
    DocumentSave.Execute(r"X:\Ansys\Geometry_Generation\Geometries_physical\geo_"+str(i)+".fmd", FileSettings1)
    # EndBlock



    # Delete Objects
    selection = Body2
    result = Delete.Execute(selection)
    # EndBlock
    
    


