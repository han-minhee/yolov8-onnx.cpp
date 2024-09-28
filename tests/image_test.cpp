// #include <gtest/gtest.h>
// #include "image_utils.hpp"

// // Example test case for Image class
// TEST(ImageTest, LoadImage0) {
//     Image image("../tests/data/image/people1.jpg");
//     EXPECT_EQ(image.getWidth(), 640);
//     EXPECT_EQ(image.getHeight(), 427);
//     EXPECT_EQ(image.getChannels(), 3);
// }

// // Resize test case
// TEST(ImageTest, ResizeImage0) {
//     Image image("../tests/data/image/people1.jpg");
//     image.resize(320, 213);
//     EXPECT_EQ(image.getWidth(), 320);
//     EXPECT_EQ(image.getHeight(), 213);
//     EXPECT_EQ(image.getChannels(), 3);
// }

// // Resize and Write test case
// TEST(ImageTest, ResizeAndWriteImage0) {
//     Image image("../tests/data/image/people1.jpg");
//     image.resize(640, 640);
//     EXPECT_EQ(image.write("people1_resized.jpg"), true);
// }