using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace HandwritingRecognition
{
    public partial class Form1 : Form
    {
        public Mat model;
        public double scale = 1.0 / 1.0;
        public Form1()
        {
            InitializeComponent();

            //String pathModel = System.IO.Directory.GetCurrentDirectory() + "\\model.png";
            //Mat source = CvInvoke.Imread(pathModel, ImreadModes.Grayscale);
            //model = new Mat();

            //CvInvoke.Resize(source, model, new Size((int) (source.Size.Width * this.scale) , (int) (source.Size.Height * this.scale)), 0, 0, Inter.Linear);
            //CvInvoke.Imshow("model", model);

            //FeatureMatching.Init(model);
        }

        private void imagebtn_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Image file (*bmp; *.jpg; *.png;*.jpeg) | *bmp; *.jpg; *.png;*.jpeg | All file (*.*) | *.*";
            if(dialog.ShowDialog() == DialogResult.OK)
            {
                using (Mat source = CvInvoke.Imread(dialog.FileName, ImreadModes.Grayscale))
                {
                    Image<Bgr, byte> imgDetected;
                    Mat transcript;
                    TranscriptDetector.Detect(source, out imgDetected, out transcript);
                    pictureBox1.Image = imgDetected.Bitmap;
                    pictureBox2.Image = transcript.Bitmap;
                }

                //Image<Bgr, byte> img = new Image<Bgr, byte>(dialog.FileName).Resize(450, 650, Emgu.CV.CvEnum.Inter.Linear);
                //Mat source = CvInvoke.Imread(dialog.FileName);
                //Mat resize = new Mat();
                //Mat gray = new Mat();
                //Mat binary = new Mat();
                //Mat canny = new Mat();

                //CvInvoke.Resize(source, resize, new Size(450, 650), 0, 0, Inter.Linear);
                //CvInvoke.CvtColor(resize, gray, ColorConversion.Bgr2Gray);
                //CvInvoke.Threshold(gray, binary, 240, 255, ThresholdType.BinaryInv);



                //CvInvoke.Imshow("image2", resize);
                //CvInvoke.Imshow("binary", binary);

                //CvInvoke.WaitKey(0);

                //long matchTime;
                //using (Mat source = CvInvoke.Imread(dialog.FileName, ImreadModes.Grayscale))
                //using (Mat sourceColor = CvInvoke.Imread(dialog.FileName, ImreadModes.Color))
                //{
                //    Mat observedImage = new Mat();
                //    Mat s_color = new Mat();
                //    CvInvoke.Resize(source, observedImage, new Size((int)(source.Size.Width * this.scale), (int)(source.Size.Height * this.scale)), 0, 0, Inter.Linear);
                //    CvInvoke.Resize(sourceColor, s_color, new Size((int)(source.Size.Width * this.scale), (int)(source.Size.Height * this.scale)), 0, 0, Inter.Linear);

                //    //Mat result = DrawMatches.Draw(this.model, observedImage, out matchTime);
                //    //CvInvoke.Imshow("KAZE - Time Match : " + matchTime.ToString(), result);

                //    VectorOfPoint Surf_result = FeatureMatching.Detect(observedImage, out matchTime);

                //    int maxX = 0, maxY = 0, minX = 10000, minY = 10000;

                //    for(int i = 0; i < Surf_result.Size; i++)
                //    {
                //        if (Surf_result[i].X > maxX) maxX = Surf_result[i].X;
                //        if (Surf_result[i].X < minX) minX = Surf_result[i].X;
                //        if (Surf_result[i].Y > maxY) maxY = Surf_result[i].Y;
                //        if (Surf_result[i].Y < minY) minY = Surf_result[i].Y;
                //    }

                //    Rectangle rect = new Rectangle(new Point(minX, minY), new Size(maxX - minX, maxY - minY));

                //    CvInvoke.Rectangle(s_color, rect, new MCvScalar(0, 255, 0));

                //    Rectangle RoiRectangle = new Rectangle(new Point((int)(minX / this.scale), (int)(minY / this.scale)), new Size((int)(rect.Width / this.scale), (int)(rect.Height / this.scale)));

                //    Mat transcript = new Mat(sourceColor, RoiRectangle);
                //    CvInvoke.Resize(transcript, transcript, new Size((int)(transcript.Width / 2), (int)(transcript.Height / 2)), 0, 0, Inter.Linear);

                //    //CvInvoke.Imshow("Transcript : ", transcript);
                //    transcript.Save("transcript.jpg");

                //    Mat binaryImage = new Mat();
                //    Mat grayImage = new Mat();

                //    CvInvoke.CvtColor(transcript, grayImage, ColorConversion.Bgr2Gray);

                //    CvInvoke.Threshold(grayImage, binaryImage, 240, 255, ThresholdType.BinaryInv);
                //    LineSegment2D [] lines = CvInvoke.HoughLinesP(binaryImage, 2, Math.PI / 2, 100, 40);

                //    List<List<LineSegment2D>> classify = new List<List<LineSegment2D>> {
                //        new List<LineSegment2D>(),
                //        new List<LineSegment2D>()
                //    };

                //    List<List<LineSegment2D>> classify_X = new List<List<LineSegment2D>> {
                //        new List<LineSegment2D>()
                //    };

                //    List<List<LineSegment2D>> classify_Y = new List<List<LineSegment2D>> {
                //        new List<LineSegment2D>()
                //    };

                //    for (int i = 0; i < lines.Length; i++)
                //    {
                //        if(Math.Abs(lines[i].P1.Y - lines[i].P2.Y) < 4)
                //        {
                //            classify[0].Add(lines[i]);
                //            //CvInvoke.Line(transcript, lines[i].P1, lines[i].P2, new MCvScalar(255, 0, 0));

                //            if(classify_X[0].Count == 0)
                //            {
                //                classify_X[0].Add(lines[i]);
                //            }
                //            else
                //            {
                //                bool has = false;
                //                for(int j = 0; j < classify_X.Count; j++)
                //                {
                //                    if(Math.Abs(lines[i].P1.Y - classify_X[j][0].P1.Y) < 10)
                //                    {
                //                        classify_X[j].Add(lines[i]);
                //                        has = true;
                //                    }
                //                }

                //                if(has == false)
                //                {
                //                    classify_X.Add(new List<LineSegment2D>
                //                    {
                //                        lines[i]
                //                    });
                //                }
                //            }
                //        }
                //        else
                //        {
                //            classify[1].Add(lines[i]);
                //            //CvInvoke.Line(transcript, lines[i].P1, lines[i].P2, new MCvScalar(0, 0, 255));

                //            if (classify_Y[0].Count == 0)
                //            {
                //                classify_Y[0].Add(lines[i]);
                //            }
                //            else
                //            {
                //                bool has = false;
                //                for (int j = 0; j < classify_Y.Count; j++)
                //                {
                //                    if (Math.Abs(lines[i].P1.X - classify_Y[j][0].P1.X) < 10)
                //                    {
                //                        classify_Y[j].Add(lines[i]);
                //                        has = true;
                //                    }
                //                }

                //                if (has == false)
                //                {
                //                    classify_Y.Add(new List<LineSegment2D>
                //                    {
                //                        lines[i]
                //                    });
                //                }
                //            }
                //        }

                        
                //    }

                //    for(int i = 0; i < classify_X.Count; i++)
                //    {
                //        CvInvoke.Line(transcript, new Point(0, classify_X[i][0].P1.Y), new Point(transcript.Size.Width, classify_X[i][0].P1.Y), new MCvScalar(0, 0, 255));
                //    }

                //    for (int i = 0; i < classify_Y.Count; i++)
                //    {
                //        CvInvoke.Line(transcript, new Point(classify_Y[i][0].P1.X, 0), new Point( classify_Y[i][0].P1.X, transcript.Size.Height), new MCvScalar(0, 255, 0));
                //    }

                //    classify_X = classify_X.OrderBy(p => p[0].P1.Y).ToList();
                //    classify_Y = classify_Y.OrderBy(p => p[0].P1.X).ToList();

                //    int k = 1;
                //    int count = 0;
                //    List<Mat> rows = new List<Mat>
                //    {
                //        new Mat(),
                //        new Mat(),
                //        new Mat()
                //    };

                //    while(classify_Y.Count - k - 1 >= 0 && count < 3)
                //    {
                //        Rectangle rect1 = new Rectangle(new Point(classify_Y[classify_Y.Count - k - 1][0].P1.X, 0), new Size(classify_Y[classify_Y.Count - k][0].P1.X - classify_Y[classify_Y.Count - k - 1][0].P1.X, transcript.Size.Height));
                //        rows[count] = new Mat(transcript, rect1);

                //        CvInvoke.Imshow("Transcript - col " + (count + 1) + " : ", rows[count]);
                //        count++;
                //        k++;
                //    }

                //    CvInvoke.Imshow("Transcript : ", transcript);
                //    CvInvoke.Imshow("Transcript - binary : ", binaryImage);
                //    CvInvoke.Imshow("SURF - Time Match : " + matchTime.ToString(), s_color);
                //}
            }
        }
    }
}
