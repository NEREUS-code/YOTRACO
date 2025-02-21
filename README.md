<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/NEREUS-code/YOTRACO">
    <img src="images/logo.png" alt="Logo" width="120" height="140">
  </a>

  <h3 align="center">YOTRACO</h3>

  <p align="center">
    Object tracking and counting system based on <a href="https://github.com/ultralytics/ultralytics">YOLO</a>
    <br />
    <br />
    <a href="https://github.com/NEREUS-code/YOTRACO/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/NEREUS-code/YOTRACO/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project leverages YOLO (You Only Look Once) for real-time object detection and tracking in video streams, enhancing analytical capabilities through automated monitoring and statistical reporting.
- **Object Detection & Tracking**: Utilizes YOLO to detect and track objects in video footage with high accuracy.
- **Crossing Detection**:  Automatically counts objects crossing a predefined line within the frame.
- **Customizable Settings**:
  - Configure the tracking line position (top, middle, bottom).
  - Define tracking direction (IN, OUT, or BOTH).
  - Select specific object classes for monitoring.
- **Processed Video Output**: Generates and saves processed video files with overlaid tracking data and object counts.
- **Logging & Statistics**: Maintains detailed records and statistical insights on tracked objects for further analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

### Installation

Install the module with pip:
```
pip install yotraco
```
**Update existing installation:** ```pip install yotraco --upgrade```\
(update as often as possible because this library is under active development)


<!-- USAGE EXAMPLES -->
### Usage

To test yotraco you can try this simple example with your video:

```python
from YOTRACO import Yotraco

model = Yotraco("yolo11l.pt",                 # the path to the yolo.pt 
                "your_video_path.mp4",        # the path to your video
                "output",                     # the name of the output
                "middle",                     # the line postion (by default : middle)
                "BOTH",                       # the track direction (by default : Both )
                classes_to_track=[0,1,2,3,4], # the class id to track 
                display=True                  # display the counts in the output video
                )

model.process_video()

```


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/NEREUS-code/YOTRACO/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NEREUS-code/YOTRACO" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT license. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Yotraco Team - nereuscode@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Thanks
Special thanks to the following resources and contributors who helped make this project possible:

- Ultralytics <a href="https://github.com/ultralytics/ultralytics">YOLO</a> for their amazing object detection framework.
- <a href="https://github.com/mouadhida"> Mouad Hida </a> for designing the project logo and enhancing the visual identity of this project.
- All contributors who provided invaluable feedback and improvements.
- The open-source community for continuous inspiration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/NEREUS-code/YOTRACO.svg?style=for-the-badge
[contributors-url]: https://github.com/NEREUS-code/YOTRACO/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/NEREUS-code/YOTRACO.svg?style=for-the-badge
[forks-url]: https://github.com/NEREUS-code/YOTRACO/network/members
[stars-shield]: https://img.shields.io/github/stars/NEREUS-code/YOTRACO.svg?style=for-the-badge
[stars-url]: https://github.com/NEREUS-code/YOTRACO/stargazers
[issues-shield]: https://img.shields.io/github/issues/NEREUS-code/YOTRACO.svg?style=for-the-badge
[issues-url]: https://github.com/NEREUS-code/YOTRACO/issues
[license-shield]: https://img.shields.io/github/license/NEREUS-code/YOTRACO.svg?style=for-the-badge
[license-url]: https://github.com/NEREUS-code/YOTRACO/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/mohammed-benyamna-504378318/

