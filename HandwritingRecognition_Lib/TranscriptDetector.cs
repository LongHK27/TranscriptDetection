using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using Emgu;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Features2D;

namespace HandwritingRecognition
{
    // (X - X0)a + (Y - Y0)b = 0
    // a.X + b.Y + c = 0
    // n(a, b)

    //Struct này không dùng
    public struct Linear
    {
        public int a;
        public int b;
        public int c;

        public Point M0;
    }

    //Class này không dùng
    public class Rank
    {
        public double value;
        public int total;
        public List<Point> items;

        public Rank(double value, Point point)
        {
            this.value = value;
            this.total = 1;
            this.items = new List<Point>() { point };
        }
    }

    // Biểu diễn đường thằng qua phương trình ax + by + c = 0, tanAlpha là hệ số góc trong phương trình y = ax + b >> tanAlpha = -a
    public class ILine
    {
        public double a;
        public double b;
        public double c;
        public double tanAlpha;
        public Point M;

        public ILine(Point A, Point B)
        {
            Point n = new Point(B.Y - A.Y, A.X - B.X);
            this.M = A;
            this.a = n.X;
            this.b = n.Y;
            this.c = -this.a * this.M.X - this.b * this.M.Y;

            this.tanAlpha = this.b != 0 ? -this.a / this.b : Double.MaxValue;
        }

        //Hàm này không dùng
        public Point FindIntersectPoint(ILine line)
        {
            var intersectPoint = new Point(Int32.MaxValue, Int32.MaxValue);

            // a1x + b1y = -c1
            // a2x + b2y = -c2

            // x = -b2y/a2 - c2 / a2 >> -a1b2y / a2 + b1y = -c1 + a1c2/a2 >> y = (c1 - a1c2/a2) / (a1b2/a2 - b1)
            // y = ( -c1 + c2 / a2 ) / (b1 - b2 * a1 / a2)

            if(line.tanAlpha != this.tanAlpha)
            {
                intersectPoint.Y = (int)((this.c - line.c * this.a / line.a) / ( -this.b + line.b * this.a / line.a));
                intersectPoint.X = (int)(-this.c - this.b * intersectPoint.Y);
            }
            return intersectPoint;
        }

        //Hàm này không dùng
        public double Distance(Point M)
        {
            return Math.Abs(this.a * M.X + this.b * M.Y + this.c) / (Math.Sqrt(Math.Pow(this.a, 2) + Math.Pow(this.b, 2)));
        }
    }

    
    
    public static class TranscriptDetector
    {
        // Tìm vị trí và cắt bảng điểm ra
        // Đầu vào là ảnh màu học bạ
        // Output:
        //      - Ảnh học bạ detectImg có vị trí của bảng điểm : hình chữ nhật màu đỏ xác định vị trí bảng điểm
        //      - Ảnh bảng điểm transcriptImage: chỉ có cột các môn và điểm của từng môn
        public static void Detect(Mat source, out Image<Bgr, byte> detectImg, out Mat transcriptImage)
        {
            Mat resizeImage = new Mat();
            Mat binaryImage = new Mat();
            Mat dilateImage = new Mat();
            Mat observedImage;

            var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(-1, -1));

            // Thay đổi kích thước ảnh ban đầu về ảnh 450x600 pixel
            CvInvoke.Resize(source, resizeImage, new Size(450, 650));

            // Chuyển sang ảnh Gray
            Image<Gray, byte> resAfter = new Image<Gray, byte>(resizeImage.Bitmap);

            // Thay đổi mức độ sáng và độ nét của ảnh
            resAfter._GammaCorrect(5.0);
            observedImage = resAfter.Clone().Mat;

            // Đưa về ảnh nhị phân
            CvInvoke.Threshold(resAfter, binaryImage, 200, 255, ThresholdType.BinaryInv);

            // Dòng này không dùng =))
            CvInvoke.Dilate(binaryImage, dilateImage, element, new Point(-1, -1), 1, BorderType.Reflect, default(MCvScalar));

            // Dùng thuật toán Hough để tìm các đường thẳng có trong ảnh
            LineSegment2D[] lines = CvInvoke.HoughLinesP(binaryImage, 2, Math.PI / 2, 50, 20);

            CvInvoke.Imshow("binaryImage asdasdsd", binaryImage);

            List<Linear> linears = new List<Linear>();
            
            // Chuyển kết quả của thuật toán Hough (đường thẳng xác định bởi 2 điểm) sang dạng phương trình ax + by + c = 0
            var listIline = new List<ILine>();

            Mat view = new Mat(resizeImage.Size, DepthType.Cv8U, 3);
            view.SetTo(new MCvScalar(0, 0, 0));

            for(int i = 0; i < lines.Length; i++)
            {
                Linear line;
                line.a = -(lines[i].P2.Y - lines[i].P1.Y);
                line.b = lines[i].P2.X - lines[i].P1.X;
                line.M0 = lines[i].P1;
                line.c = line.a * (-line.M0.X) + line.b * (-line.M0.Y);

                linears.Add(line);

                // create linear;
                var iline = new ILine(lines[i].P1, lines[i].P2);
                listIline.Add(iline);
            }

            // Phân loại các đường: Nếu khoảng cách giữa 2 đường quá nhỏ thì nhóm chung lại 1 nhóm
            var classifyILine = DBSCANofILine(listIline, 8);
            
            //Vẽ các đường lên ảnh
            for(var i = 0; i < classifyILine.Count; i++)
            {
                var A = new Point();
                var B = new Point();

                if(classifyILine[i][0].tanAlpha == 0)
                {
                    //A = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(0, 0), new Point(0, view.Size.Height)));
                    //B = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(view.Size.Width, 0), new Point(view.Size.Width, view.Size.Height)));

                    A.Y = classifyILine[i][0].M.Y;
                    B.X = view.Size.Width;
                    B.Y = A.Y;
                }

                if(classifyILine[i][0].tanAlpha == Double.MaxValue)
                {
                    //A = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(0, 0), new Point(view.Size.Width, 0)));
                    //B = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(0, view.Size.Height), new Point(view.Size.Width, view.Size.Height)));

                    A.X = classifyILine[i][0].M.X;
                    B.X = A.X;
                    B.Y = view.Size.Height;
                }

                //if(classifyILine[i][0].tanAlpha < Math.Tan(Math.PI / 6))
                CvInvoke.Line(resAfter, A, B, new MCvScalar(0, 0, 0), 1);
                
            }

            // Phân loại các đường:
            //      - list_2[0] : các đường thẳng song song Ox
            //      - list_2[1] : các đường thẳng song song Oy
            var list_2 = ClassifyXY(classifyILine);
            
            // Tìm các đường thẳng song song có khoảng cách gần như không đổi (khoảng cách các đường thẳng phân chia các môn trong bảng điểm)
            var startIndex = 0;
            var endIndex = 0;
            var step = 0.0;
            var count = 0;

            var currentStartIndex = 0;
            var currentEndIndex = 0;
            var currentStep = 0.0;
            var currentCount = 0;

            for(var i = 0; i < list_2[0].Count - 1; i++)
            {
                if(i == 0 || Math.Abs(currentStep - list_2[0][i].Distance(list_2[0][i + 1].M)) < 5)
                {
                    if (i == 0) currentStep = list_2[0][i].Distance(list_2[0][i + 1].M);
                    currentCount++;
                    currentEndIndex++;
                    if(currentCount > count)
                    {
                        startIndex = currentStartIndex;
                        endIndex = currentEndIndex;
                        count = currentCount;
                        step = currentStep;
                    }
                }
                else
                {
                    currentStartIndex = i;
                    currentEndIndex = i;
                    currentCount = 0;
                    currentStep = list_2[0][i].Distance(list_2[0][i + 1].M);
                }
            }

            //if(startIndex > 1) startIndex -= 2;
            endIndex++;

            // Đoạn này không dùng : Debug để xem có lấy đúng các đường thẳng của bảng hay không
            for(var i = startIndex; i <= endIndex; i++)
            {
                var A = new Point();
                var B = new Point();

                if (list_2[0][i].tanAlpha == 0)
                {
                    //A = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(0, 0), new Point(0, view.Size.Height)));
                    //B = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(view.Size.Width, 0), new Point(view.Size.Width, view.Size.Height)));

                    A.Y = list_2[0][i].M.Y;
                    B.X = view.Size.Width;
                    B.Y = A.Y;
                }

                if (list_2[0][i].tanAlpha == Double.MaxValue)
                {
                    //A = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(0, 0), new Point(view.Size.Width, 0)));
                    //B = classifyILine[i][0].FindIntersectPoint(new ILine(new Point(0, view.Size.Height), new Point(view.Size.Width, view.Size.Height)));

                    A.X = list_2[0][i].M.X;
                    B.X = A.X;
                    B.Y = view.Size.Height;
                }

                if (list_2[0][i].tanAlpha < Math.Tan(Math.PI / 6))
                    CvInvoke.Line(view, A, B, new MCvScalar(255, 255, 255), 1);
            }

            
            // Cắt toàn bộ bảng điểm: chỉ có giới hạn bên trên và bên dưới - ảnh vẫn chứa phần ghi chú và phần chữ ký giáo viên
            var rectTransciptX = new Rectangle(new Point(0, list_2[0][startIndex].M.Y), new Size(resizeImage.Size.Width, list_2[0][endIndex].M.Y - list_2[0][startIndex].M.Y));

            var rectTransciptXImg = new Mat(binaryImage, rectTransciptX);

            //using (var model = CvInvoke.Imread(System.IO.Directory.GetCurrentDirectory() + "\\model-3.png", ImreadModes.Grayscale))
            //{
            //    //FeatureMatching.Init(model);
            //    //var transcriptXX = new Mat(observedImage, rectTransciptX);
            //    //long matchTime;

            //    //VectorOfPoint pointsMatching = FeatureMatching.Detect(transcriptXX, out matchTime);

            //    //Mat r = DrawMatches.Draw(model, transcriptXX, out matchTime);
            //    //CvInvoke.Imshow("imgasd", r);

            //    //CvInvoke.Polylines(transcriptXX, pointsMatching.ToArray(), true, new MCvScalar(0, 0, 0), 2);

            //    //Point[] listPoint = pointsMatching.ToArray().OrderBy(x => x.X).ToArray();

            //    //CvInvoke.Imshow("tran", transcriptXX);


            //}
            
            

            // Cắt phần bảng điểm
            
            // Dùng các đặc trưng của ảnh sẽ phân ra đc các vùng: tách vùng bảng điểm với vùng chữ ký
            // Kết hợp với các đường thẳng song song Oy trong list_2[1] để lấy vị trí của bảng điểm
            
            using (var transcriptBefore = new Mat(observedImage, rectTransciptX))
            using (var keyPointImage = new Image<Bgr, byte>(transcriptBefore.Size.Width, transcriptBefore.Size.Height))
            {
                //Tìm đặc trưng của ảnh rồi vẽ lên ảnh nhị phân khác
                var keyPoints = FeatureMatching.FindFeature(transcriptBefore);

                keyPointImage.SetValue(new MCvScalar(0, 0, 0));

                for (var i = 0; i < keyPoints.Size; i++)
                {
                    CvInvoke.Circle(keyPointImage, new Point((int)keyPoints[i].Point.X, (int)keyPoints[i].Point.Y), 2, new MCvScalar(255, 255, 255), 2);
                }
                var histogramY = new List<int>();

                for (var i = 0; i < keyPointImage.Size.Width; i++)
                {
                    histogramY.Add(0);
                    for (var j = 0; j < keyPointImage.Size.Height; j++)
                    {
                        if (keyPointImage.Data[j, i, 0] + keyPointImage.Data[j, i, 1] + keyPointImage.Data[j, i, 2] > 0)
                        {
                            histogramY[i]++;
                        }
                    }
                }

                var keyPointBinary = new Mat(keyPointImage.Size, DepthType.Cv8U, 1);
                CvInvoke.CvtColor(keyPointImage, keyPointBinary, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(keyPointBinary, keyPointBinary, 100, 255, ThresholdType.Binary);

                var avg = 80;

                var start_transcript = 0;

                for (var i = 0; i < histogramY.Count; i++)
                {
                    if (histogramY[i] > avg)
                    {

                        for (var j = 0; j < list_2[1].Count; j++)
                        {
                            if (Math.Abs(list_2[1][j].M.X - i) < 20)
                            {
                                start_transcript = j;
                                CvInvoke.Rectangle(keyPointBinary, new Rectangle(new Point(0, 0), new Size(list_2[1][j].M.X, keyPointBinary.Size.Height)), new MCvScalar(0, 0, 0), -1);
                                CvInvoke.Rectangle(keyPointImage, new Rectangle(new Point(0, 0), new Size(list_2[1][j].M.X, keyPointBinary.Size.Height)), new MCvScalar(0, 0, 0), -1);
                                CvInvoke.Line(keyPointImage, new Point(list_2[1][j].M.X, 0), new Point(list_2[1][j].M.X, keyPointImage.Size.Height), new MCvScalar(255, 0, 0), 3);
                                break;
                            }
                        }
                        break;
                    }
                }

                var count_area = 0;

                for (var i = start_transcript; i < list_2[1].Count - 1; i++)
                {
                    CvInvoke.Circle(keyPointImage, new Point(list_2[1][i].M.X, list_2[0][startIndex + 3].M.Y - rectTransciptX.Y), 3, new MCvScalar(0, 0, 255), 2);
                    CvInvoke.Circle(keyPointImage, new Point(list_2[1][i].M.X, list_2[0][startIndex + 4].M.Y - rectTransciptX.Y), 3, new MCvScalar(0, 0, 255), 2);
                    var rect_area = new Rectangle(new Point(list_2[1][i].M.X, list_2[0][startIndex + 3].M.Y - rectTransciptX.Y), new Size(list_2[1][i + 1].M.X - list_2[1][i].M.X, list_2[0][startIndex + 4].M.Y - list_2[0][startIndex + 3].M.Y));
                    CvInvoke.Rectangle(keyPointImage, rect_area, new MCvScalar(255, 255, 00), 2);
                    using (var rectImg2 = new Mat(binaryImage, new Rectangle(new Point(list_2[1][i].M.X + 2, list_2[0][startIndex + 3].M.Y + 2), new Size(list_2[1][i + 1].M.X - list_2[1][i].M.X - 4, list_2[0][startIndex + 4].M.Y - list_2[0][startIndex + 3].M.Y - 4))))
                    {
                        var nonZR = (double)CvInvoke.CountNonZero(rectImg2) / (rectImg2.Size.Width * rectImg2.Size.Height);

                        if (nonZR < 0.02)
                        {
                            count_area = i;
                            break;
                        }

                    }

                }

                CvInvoke.Line(keyPointImage, new Point(list_2[1][count_area].M.X, 0), new Point(list_2[1][count_area].M.X, keyPointImage.Size.Height), new MCvScalar(255, 0, 0), 3);

                // Cắt lấy ảnh bảng điểm có kích thước ban đầu
                var transcriptColor = new Image<Bgr, byte>(resAfter.Bitmap);

                CvInvoke.Rectangle(transcriptColor, new Rectangle(new Point(list_2[1][start_transcript].M.X, list_2[0][startIndex - 2].M.Y), new Size(list_2[1][count_area].M.X - list_2[1][start_transcript].M.X, list_2[0][endIndex].M.Y - list_2[0][startIndex - 2].M.Y)), new MCvScalar(0, 0, 255), 2);
                CvInvoke.Imshow("Transcript Detected", transcriptColor);


                for (var i = 0; i < classifyILine.Count; i++)
                {
                    var A = new Point();
                    var B = new Point();

                    CvInvoke.Line(transcriptColor, A, B, new MCvScalar(0, 0, 255), 1);

                }

                CvInvoke.Imshow("transcriptColor", transcriptColor);

                detectImg = transcriptColor.Clone();

                double scaleX = source.Size.Width / 450 * 1.2;
                double scaleY = source.Size.Height / 650 * 1.2;

                var rect_scale = new Rectangle(
                    new Point((int)(list_2[1][start_transcript].M.X * scaleX), (int)(list_2[0][startIndex - 2].M.Y * scaleY)),
                    new Size((int)((list_2[1][count_area].M.X - list_2[1][start_transcript].M.X) * scaleX), (int)((list_2[0][endIndex].M.Y - list_2[0][startIndex - 2].M.Y) * scaleY)));

                var rect_scale_transcript = new Rectangle(
                    new Point((int)(list_2[1][start_transcript].M.X * scaleX), (int)(list_2[0][startIndex].M.Y * scaleY)),
                    new Size((int)((list_2[1][count_area].M.X - list_2[1][start_transcript].M.X) * scaleX), (int)((list_2[0][endIndex].M.Y - list_2[0][startIndex].M.Y) * scaleY)));

                var transcrip_scale = new Mat(source, rect_scale);
                var transcript_t = new Mat(source, rect_scale_transcript);

                var bfff = transcrip_scale.ToImage<Bgr, byte>();
                bfff._GammaCorrect(5.0);

                var transcipt_origin = transcript_t.ToImage<Bgr, byte>();
                transcipt_origin._GammaCorrect(5.0);
                
                
                var line_on_transcript = FindLine(transcipt_origin.Mat, 100, 50);

                var lineX = line_on_transcript[0];
                var lineY = line_on_transcript[1];
                var random = new Random();
                for (var i = 1; i < 10; i++) // toan, ly, hoa, sinh, tin, van, su, dia, TA
                {

                    var p1 = new Point(lineY[lineY.Count - 3].M.X, lineX[i].M.Y);
                    var p2 = new Point(lineY[lineY.Count - 2].M.X, lineX[i].M.Y);
                    var p3 = new Point(lineY[lineY.Count - 1].M.X, lineX[i].M.Y);

                    var p4 = new Point(lineY[lineY.Count - 3].M.X, lineX[i - 1].M.Y);
                    var p5 = new Point(lineY[lineY.Count - 2].M.X, lineX[i - 1].M.Y);
                    var p6 = new Point(lineY[lineY.Count - 1].M.X, lineX[i - 1].M.Y);

                    //hk 1
                    //CvInvoke.Circle(transcipt_origin, p1, 6, new MCvScalar(255,0,0), 2);
                    var pic1 = new Mat(transcipt_origin.Mat, new Rectangle(new Point(p4.X + 10, p4.Y + 5), new Size(p2.X - p1.X - 20, p1.Y - p4.Y - 8)));
                    //CvInvoke.Imshow("mark1 " + i, pic1);
                    DetectDigit(pic1, "mark1 - " + i + " - " + random.Next());
                    //hk 2
                    //CvInvoke.Circle(transcipt_origin, p2, 6, new MCvScalar(255, 0, 0), 2);
                    var pic2 = new Mat(transcipt_origin.Mat, new Rectangle(new Point(p5.X + 10, p5.Y + 5), new Size(p3.X - p2.X - 20, p2.Y - p5.Y - 8)));
                    //CvInvoke.Imshow("mark2 " + i, pic2);
                    DetectDigit(pic2, "mark2 - " + i + " - " + random.Next());
                    //CN
                    //CvInvoke.Circle(transcipt_origin, p3, 6, new MCvScalar(255, 0, 0), 2);
                    var pic3 = new Mat(transcipt_origin.Mat, new Rectangle(new Point(p6.X + 10, p6.Y + 5), new Size(transcipt_origin.Size.Width - p3.X - 20, p3.Y - p6.Y - 8)));
                    CvInvoke.Imshow("mark3 " + i, pic3);
                    
                    DetectDigit(pic3, "mark3 - " + i + " - " + random.Next());
                    
                }

                transcriptImage = transcipt_origin.Mat;
                CvInvoke.Imshow("transcipt_origin", transcriptImage);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static int DetectDigit(Mat source, string filename)
        {
            var binaryImg = new Mat();
            var grayImg = new Mat();
            var edgeImg = new Mat();

            CvInvoke.CvtColor(source, grayImg, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(grayImg, binaryImg, 150, 255, ThresholdType.BinaryInv);
            var element = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(-1, -1));

            CvInvoke.Dilate(binaryImg, binaryImg, element, new Point(-1, -1), 1, BorderType.Reflect, default(MCvScalar));

            //CvInvoke.Imwrite("E:\\ReadCV\\image" + filename + ".jpg", binaryImg);
            CvInvoke.Canny(binaryImg, edgeImg, 100, 255);
            //CvInvoke.Imwrite("E:\\ReadCV\\image\\canny-" + filename + ".jpg", edgeImg);
            SaveGray(binaryImg, filename);
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                //var digit = new Mat(source, new Rectangle(new Point(0, 0), source.Size));
                var digit = source.Clone();
                digit.SetTo(new MCvScalar(0, 0, 0));

                Mat hierarchy = new Mat();

                CvInvoke.FindContours(edgeImg, contours, hierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

                var listRect = new List<Rectangle>();

                for (int i = 0; i < contours.Size; i++)
                {
                    Point[] pointOfContour = contours[i].ToArray();

                    var minx = Int32.MaxValue;
                    var miny = Int32.MaxValue;
                    var maxx = 0;
                    var maxy = 0;

                    for(var j = 0; j < pointOfContour.Length; j++)
                    {
                        if (pointOfContour[j].X < minx) minx = pointOfContour[j].X;
                        if (pointOfContour[j].Y < miny) miny = pointOfContour[j].Y;
                        if (pointOfContour[j].X > maxx) maxx = pointOfContour[j].X;
                        if (pointOfContour[j].X > maxy) maxy = pointOfContour[j].X;
                    }

                    var r = new Rectangle(new Point(minx, miny), new Size(maxx - minx, maxy - miny));
                    CvInvoke.Rectangle(source, r, new MCvScalar(0, 0, 255), 1);
                    listRect.Add(r);

                    CvInvoke.DrawContours(digit, contours, i, new MCvScalar(255, 255, 255));

                    //using (VectorOfVectorOfPoint contours1 = new VectorOfVectorOfPoint())
                    //{
                    //    CvInvoke.Threshold(digit, digit, 50, 255, ThresholdType.Binary);
                    //    Mat hierarchy1 = new Mat();

                    //    var e = new Image<Gray, byte>(digit.Bitmap);

                    //    CvInvoke.Imshow("e", e);

                    //    CvInvoke.FindContours(e, contours1, hierarchy1, RetrType.Tree, ChainApproxMethod.ChainApproxNone);

                    //    for (int k = 0; k < contours1.Size; k++)
                    //    {
                    //        Point[] pointOfContour1 = contours1[k].ToArray();

                    //        var minx1 = Int32.MaxValue;
                    //        var miny1 = Int32.MaxValue;
                    //        var maxx1 = 0;
                    //        var maxy1 = 0;

                    //        for (var g = 0; g < pointOfContour1.Length; g++)
                    //        {
                    //            if (pointOfContour1[g].X < minx1) minx1 = pointOfContour1[g].X;
                    //            if (pointOfContour1[g].Y < miny1) miny1 = pointOfContour1[g].Y;
                    //            if (pointOfContour1[g].X > maxx1) maxx1 = pointOfContour1[g].X;
                    //            if (pointOfContour1[g].X > maxy1) maxy1 = pointOfContour1[g].X;
                    //        }

                    //        var r1 = new Rectangle(new Point(minx1, miny1), new Size(maxx1 - minx1, maxy1 - miny1));
                    //        CvInvoke.Rectangle(source, r1, new MCvScalar(0, 0, 255), -1);
                    //    }
                    //}
                    //CvInvoke.Imwrite("E:\\ReadCV\\image\\" + filename + ".jpg", digit);

                    CvInvoke.Imshow("edge - contours", digit);
                }

                listRect = listRect.OrderByDescending(x => x.Width * x.Height).ToList();
                //CvInvoke.Rectangle(source, listRect[0], new MCvScalar(0, 0, 255));
                //if (listRect.Count > 1) CvInvoke.Rectangle(source, listRect[1], new MCvScalar(0, 0, 255));
                //if (listRect.Count > 2) CvInvoke.Rectangle(source, listRect[2], new MCvScalar(0, 0, 255));
                //if (listRect.Count > 3) CvInvoke.Rectangle(source, listRect[3], new MCvScalar(0, 0, 255));
            }

            CvInvoke.Imshow("binary - ad", binaryImg);
            CvInvoke.Imshow("edge", edgeImg);
            CvInvoke.Imshow("contours", source);

            return 12;
        }

        public static void SaveGray(Mat source, string filename)
        {
            var bff = source.Clone();
            bff.SetTo(new MCvScalar(0, 0, 0));
            var resultImage = bff.ToImage<Gray, byte>();
            var sourceCopy = source.ToImage<Gray, byte>();

            for(var i = 0; i < source.Width; i++)
            {
                var sum = 0;
                for(var j = 0; j < source.Height; j++)
                {
                    if (sourceCopy.Data[j, i, 0] > 0) sum++;
                }

                for (var j = 0; j < source.Height; j++)
                {
                    if(sum > 10) resultImage.Data[j, i, 0] = (byte)(sum * 4);
                }
            }

            //CvInvoke.Imwrite("E:\\ReadCV\\gray\\" + filename + ".jpg", resultImage);

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="cannyEdges"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static VectorOfPoint FindLargestContour(IInputOutputArray cannyEdges, IInputOutputArray result)
        {
            int largest_contour_index = 0;
            double largest_area = 100;
            VectorOfPoint largestContour;

            using (Mat hierachy = new Mat())
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                IOutputArray hirarchy;

                CvInvoke.FindContours(cannyEdges, contours, hierachy, RetrType.Tree, ChainApproxMethod.ChainApproxNone);

                for (int i = 0; i < contours.Size; i++)
                {
                    MCvScalar color = new MCvScalar(0, 0, 255);

                    double a = CvInvoke.ContourArea(contours[i], false);  //  Find the area of contour
                    if (a > largest_area)
                    {
                        largest_area = a;
                        largest_contour_index = i;                //Store the index of largest contour
                    }

                    CvInvoke.DrawContours(result, contours, largest_contour_index, new MCvScalar(255, 0, 0));
                }

                CvInvoke.DrawContours(result, contours, largest_contour_index, new MCvScalar(0, 0, 255), 3, LineType.EightConnected, hierachy);
                largestContour = new VectorOfPoint(contours[largest_contour_index].ToArray());
            }

            return largestContour;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public static List<List<ILine>> FindLine(Mat source, int minLength, int miniBatchSize)
        {
            var binaryImage = new Mat();
            Image<Gray, byte> grayImage = new Image<Gray, byte>(source.Bitmap);

            CvInvoke.Threshold(grayImage, binaryImage, 200, 255, ThresholdType.BinaryInv);

            LineSegment2D[] lines = CvInvoke.HoughLinesP(binaryImage, 2, Math.PI / 2, 50, minLength);

            List<Linear> linears = new List<Linear>();
            var listIline = new List<ILine>();

            for (int i = 0; i < lines.Length; i++)
            {
                Linear line;
                line.a = -(lines[i].P2.Y - lines[i].P1.Y);
                line.b = lines[i].P2.X - lines[i].P1.X;
                line.M0 = lines[i].P1;
                line.c = line.a * (-line.M0.X) + line.b * (-line.M0.Y);

                linears.Add(line);

                // create linear;
                var iline = new ILine(lines[i].P1, lines[i].P2);
                listIline.Add(iline);
            }

            var classifyILine = DBSCANofILine(listIline, miniBatchSize);
            for (var i = 0; i < classifyILine.Count; i++)
            {
                var A = new Point();
                var B = new Point();

                if (classifyILine[i][0].tanAlpha == 0)
                {
                    A.Y = classifyILine[i][0].M.Y;
                    B.X = source.Size.Width;
                    B.Y = A.Y;
                }

                if (classifyILine[i][0].tanAlpha == Double.MaxValue)
                {
                    A.X = classifyILine[i][0].M.X;
                    B.X = A.X;
                    B.Y = source.Size.Height;
                }

                //CvInvoke.Line(source, A, B, new MCvScalar(0, 0, 255), 2);
            }

            return ClassifyXY(classifyILine);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static List<List<ILine>> ClassifyXY(List<List<ILine>> list)
        {
            var result = new List<List<ILine>>
            {
                new List<ILine>(),  /*--- f(x) // Ox ---*/
                new List<ILine>()   /*--- f(x) // Oy ---*/
            };
            
            for(var i = 0; i < list.Count; i++)
            {
                if(list[i][0].tanAlpha < Math.Tan(Math.PI / 4))
                {
                    result[0].Add(list[i][0]);
                }
                else
                {
                    result[1].Add(list[i][0]);
                }
            }

            result[0] = result[0].OrderBy(x => x.M.Y).ToList(); // f(x) // Ox >> sort by Y
            result[1] = result[1].OrderBy(x => x.M.X).ToList(); // f(x) // Ox >> sort by X

            return result;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static List<List<ILine>> DBSCANofILine(List<ILine> list, int miniBatchSize)
        {
            var result = new List<List<ILine>>();

            for(var i = 0; i < list.Count; i++)
            {
                if(result.Count == 0)
                {
                    var list_i = new List<ILine>();
                    list_i.Add(list[i]);
                    result.Add(list_i);
                }
                else
                {
                    var isHas = false;
                    for(var j = 0; j < result.Count; j++)
                    {
                        if(list[i].a == 0 && result[j][0].tanAlpha == list[i].tanAlpha && Math.Abs(list[i].M.Y - result[j][0].M.Y) < miniBatchSize)
                        {
                            result[j].Add(list[i]);
                            isHas = true;
                            break;
                        }

                        if (list[i].b == 0 && result[j][0].tanAlpha == list[i].tanAlpha && Math.Abs(list[i].M.X - result[j][0].M.X) < miniBatchSize)
                        {
                            result[j].Add(list[i]);
                            isHas = true;
                            break;
                        }
                    }

                    if(!isHas)
                    {
                        var list_i = new List<ILine>();
                        list_i.Add(list[i]);
                        result.Add(list_i);
                    }
                }
            }

            result = result.OrderByDescending(x => x.Count).ToList();

            for(var i = 0; i < result.Count; i++)
            {
                if(result[i].Count > 5)
                {
                    result.Remove(result[i]);
                }
            }

            return result;
        }


        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="list"></param>
        /// <param name="rank"></param>
        /// <returns></returns>
        public static double[,] CalculateDistance(List<Linear> list, out List<Rank> rank)
        {
            double[,] dis = new double[list.Count, list.Count];
            rank = new List<Rank>();

            for(int i = 0; i < list.Count; i++)
            {
                dis[i, i] = 0;
                for(int j = i + 1; j < list.Count; j++)
                {
                    dis[i, j] = Math.Abs(list[i].a * list[j].M0.X + list[i].b * list[j].M0.Y + list[i].c) / Math.Sqrt(list[i].a * list[i].a + list[i].b * list[i].b);
                    dis[j, i] = dis[i, j];

                    
                    if(rank.Count == 0)
                    {
                        Rank r = new Rank(dis[i, j], new Point(i, j));
                        rank.Add(r);
                    } else
                    {
                        bool has = false;
                        for (int k = 0; k < rank.Count; k++)
                        {
                            if (rank[k].value == dis[i, j])
                            {
                                rank[k].items.Add(new Point(i, j));
                                rank[k].total++;
                                has = true;
                                break;
                            }
                        }

                        if(!has)
                        {
                            Rank r = new Rank(dis[i, j], new Point(i, j));
                            rank.Add(r);
                        }
                    }
                    
                }
            }

            return dis;
        }

        public static List<List<Linear>> DBSCAN_X (List<Linear> lines)
        {
            List<List<Linear>> list = new List<List<Linear>>();

            for(int i = 0; i < lines.Count; i++)
            {
                if(list.Count == 0)
                {
                    List<Linear> line_n = new List<Linear>();
                    line_n.Add(lines[i]);
                    list.Add(line_n);
                }
                else
                {
                    bool has = false;

                    for (int j = 0; j < list.Count; j++)
                    {
                        if (lines[i].a == list[j][0].a && lines[i].b == list[j][0].b && lines[i].c == list[j][0].c)
                        {
                            list[j].Add(lines[i]);
                            has = true;
                            break;
                        }
                    }

                    if(!has)
                    {
                        List<Linear> line_n = new List<Linear>();
                        line_n.Add(lines[i]);
                        list.Add(line_n);
                    }
                }
            }

            return list;
        }
        

    }
}
