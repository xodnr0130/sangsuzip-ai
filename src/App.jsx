import React, { useRef, useEffect } from "react";
import * as posenet from "@tensorflow-models/posenet";
import { drawBoundingBox, drawKeypoints, drawSkeleton } from "./utils/posenet";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { styled } from "styled-components";

const App = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const textRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const text = textRef.current;
    const context = canvas.getContext("2d");

    let checkRvalue = [];
    let checkHSSCValue = [];
    let checkOverlapValue = [];
    let prevHeadPosX = 0.0;
    let prevHeadPosY = 0.0;

    const loadModelAndPredict = async () => {
      // 포즈넷 모델 로드
      const model = await posenet.load({
        architecture: "MobileNetV1",
      });

      // 비디오 스트리밍 및 예측 시작
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;

      video.onloadeddata = () => {
        predict(model);
      };
    };

    const gru_predict = async (
      checkRvalue,
      checkHSSCValue,
      checkOverlapValue
    ) => {
      // model json으로 export한 이름 맞춰줘야 함
      // 딥러닝 모델 load
      const gruModel = await tf.loadLayersModel("/model.json");
      // checkRvalue -> newRList

      // ,기준 으로 분할해서 rList 생성
      let rList = checkRvalue.split(",");
      let newRList = [];

      // 데이터가 이런식으로 넘어옴
      // [[null,0.18657834490012104,0.18657834490012104,0.18618795259034135,0.18344568199460792,0.19849944542954095,0.1974836802320571,0.1974836802320571,0.22487983991353605,0.21855182462097844,0.21320347806423676,0.1924610105813943,0.18501344863201524,0.19023769931997003,0.28993111227569507,0.40117841057478293],[null, 0.18657834490012104,0.18657834490012104,0.18618795259034135,0.18344568199460792,0.19849944542954095,0.1974836802320571,0.1974836802320571,0.22487983991353605,0.21855182462097844,0.21320347806423676,0.1924610105813943,0.18501344863201524,0.19023769931997003,0.28993111227569507,0.40117841057478293]]

      // rList 길이가 33이면
      if (rList.length == 33) {
        // 첫번째거 잘라내고
        rList = rList.slice(1);
        // 마지막거 2개 제외해서 rList 만듬 -> 30개로 맞추기
        rList = rList.slice(0, rList.length - 2);
      } else {
        // 여기도 30개로 맞추기 위해서 야매로 조건문으로 처리
        // 한 프레임에 요소 하나씩 들어가야 하는데, 2개가 들어갈 때도 있어서
        // 33개나 32개여서 이렇게 야매로 처리함
        rList = rList.slice(1);
        rList = rList.slice(0, rList.length - 1);
      }

      let i = rList.indexOf("[null");
      delete rList[i];
      delete rList[i - 1];

      // 요소 값들 소수점 아래 세자리로 만들어서 newRList로 변환
      for (var rl in rList) {
        var real = parseFloat(rList[rl]).toFixed(3);
        newRList.push(real);
      }
      console.log("newRList", newRList);

      // newRList를 만드는 이 과정을 newHsscList와 newOverlapList 만들 때 똑같이 함

      // checkHSSCValue -> newHsscList
      let hsscList = checkHSSCValue.split(",");
      let newHsscList = [];

      if (hsscList.length == 32) {
        hsscList = hsscList.slice(1);
        hsscList = hsscList.slice(0, hsscList.length - 1);
      } else {
        hsscList = hsscList.slice(1);
      }

      i = hsscList.indexOf("[null");
      delete hsscList[i];
      delete hsscList[i - 1];

      for (var hl in hsscList) {
        var real = parseFloat(hsscList[hl]).toFixed(3);
        newHsscList.push(real);
      }

      console.log("newHsscList", newHsscList);

      // checkOverlapValue -> newOverlapList
      let overlapList = checkOverlapValue.split(",");
      let newOverlapList = [];

      if (overlapList.length == 33) {
        overlapList = overlapList.slice(1);
        overlapList = overlapList.slice(0, overlapList.length - 2);
      } else {
        overlapList = overlapList.slice(1);
        overlapList = overlapList.slice(0, overlapList.length - 1);
      }

      i = overlapList.indexOf("[null");
      delete overlapList[i];
      delete overlapList[i - 1];

      for (var ol in overlapList) {
        var real = parseFloat(overlapList[ol]).toFixed(3);
        newOverlapList.push(real);
      }

      console.log("newOverlapList", newOverlapList);

      let merged = newRList.concat(newHsscList);
      merged = merged.concat(newOverlapList);
      console.log("merged", merged);

      // model에 넣어줄 tensor [1batch size, 30frame, 3features], datatype = float 32bit
      let x = tf.tensor(merged, [1, 30, 3], "float32");

      // load한 모델에 input data = x 넣어서 predict
      const prediction = gruModel.predict(x);
      console.log("prediction", prediction);

      // dataSync(): tensor -> Javascript 배열로 반환
      // 예측 결과 배열 반환
      const predictionArray = prediction.dataSync();
      console.log("predictionArray", predictionArray);

      // 최대값 구해서 maxValue에 저장
      const maxValue = Math.max(...predictionArray);
      console.log("maxValue", maxValue);

      // 낙상 판단 0.5 기준으로
      if (maxValue > 0.5) {
        text.innerText = "낙상이 감지되었습니다!";
      } else {
        text.innerText = "낙상이 감지되지 않았습니다!";
      }
    };

    const predict = async (model) => {
      if (checkOverlapValue.length % 32 === 0) {
        gru_predict(
          JSON.stringify(checkRvalue),
          JSON.stringify(checkHSSCValue),
          JSON.stringify(checkOverlapValue)
        );
        checkRvalue = [];
        checkHSSCValue = [];
        checkOverlapValue = [];
      }
      // 포즈넷 모델로 포즈 추정
      const pose = await model.estimateSinglePose(video);
      canvas.width = video.width;
      canvas.height = video.height;

      console.log("Keypoints JSON", JSON.stringify(pose.keypoints));

      context.clearRect(0, 0, canvas.width, canvas.height);

      drawKeypoints(pose.keypoints, 0.1, context);
      drawSkeleton(pose.keypoints, 0.1, context);

      let totalHeadX = 0.0;
      let totalHeadY = 0.0;

      let head = [];
      for (let i = 0; i < 7; i++) {
        head.push(pose.keypoints[i]);
        totalHeadX += pose.keypoints[i].position.x;
        totalHeadY += pose.keypoints[i].position.y;
      }
      let body = [];
      for (let i = 5; i < 13; i++) {
        if (i == 7 || i == 8 || i == 9 || i == 10) continue;
        body.push(pose.keypoints[i]);
      }
      let leg = [];
      for (let i = 11; i < 17; i++) {
        leg.push(pose.keypoints[i]);
      }

      const headBoundingBox = posenet.getBoundingBox(head);
      //body_boundingbox = posenet.getBoundingBox(body);
      const bodyBoundingBox = posenet.getBoundingBox(body);
      const headArea =
        (headBoundingBox.maxX - headBoundingBox.minX) *
        (headBoundingBox.maxY - headBoundingBox.minY);
      if (headBoundingBox.maxY > bodyBoundingBox.minY) {
        const overlapWidth = Math.max(
          0,
          Math.min(
            headBoundingBox.minX +
              (headBoundingBox.maxX - headBoundingBox.minX),
            bodyBoundingBox.minX + (bodyBoundingBox.maxX - bodyBoundingBox.minX)
          ) - Math.max(headBoundingBox.minX, bodyBoundingBox.minX)
        );
        const overlapHeight = Math.max(
          0,
          Math.min(
            headBoundingBox.minY +
              (headBoundingBox.maxY - headBoundingBox.minY),
            bodyBoundingBox.minY + (bodyBoundingBox.maxY - bodyBoundingBox.minY)
          ) - Math.max(headBoundingBox.minY, bodyBoundingBox.minY)
        );
        const overlapArea = overlapWidth * overlapHeight;
        checkOverlapValue.push(overlapArea);
      } else {
        const overlapArea = 0.0;
        checkOverlapValue.push(overlapArea);
      }

      const legBoundingbox = posenet.getBoundingBox(leg);

      if (headBoundingBox.minY < bodyBoundingBox.maxY + 10) {
        console.log("머리, 몸통 segment 중첩!");
      }

      if (bodyBoundingBox.minY < legBoundingbox.maxY + 10) {
        console.log("몸통, 다리 segment 중첩!");
      }

      drawBoundingBox(head, context);
      drawBoundingBox(body, context);
      drawBoundingBox(leg, context);

      if (prevHeadPosX == 0 && prevHeadPosY == 0) {
        prevHeadPosX = totalHeadX / 7;
        prevHeadPosY = totalHeadY / 7;
      } else {
        let posneg = 1;
        if (prevHeadPosY < totalHeadY / 7) posneg = -1;
        const distance = Math.sqrt(
          Math.pow(totalHeadX / 7 - prevHeadPosX, 2) +
            Math.pow(totalHeadY / 7 - prevHeadPosY, 2)
        );
        const speed = posneg * (distance / 33);
        checkHSSCValue.push(speed);
        prevHeadPosX = totalHeadX / 7;
        prevHeadPosY = totalHeadY / 7;
      }

      const boundingBox = posenet.getBoundingBox(pose.keypoints);
      console.log(
        "R값 : ",
        (boundingBox.maxX - boundingBox.minX) /
          (boundingBox.maxY - boundingBox.minY)
      );
      checkRvalue.push(
        (boundingBox.maxX - boundingBox.minX) /
          (boundingBox.maxY - boundingBox.minY).toFixed(2)
      );

      // gru 모델 호출 및 예측
      gru_predict(checkRvalue, checkHSSCValue, checkOverlapValue);

      // 예측 함수 재귀 호출
      requestAnimationFrame(() => predict(model));
    };

    loadModelAndPredict();
  }, []);

  return (
    <Container>
      <Video
        ref={videoRef}
        width="800"
        height="600"
        muted
        autoPlay
        playsInline
      ></Video>
      <Canvas ref={canvasRef}></Canvas>
      <Text ref={textRef}></Text>
    </Container>
  );
};

export default App;

const Container = styled.div``;

const Video = styled.video`
  position: absolute;
  transform: translate(-50%, -50%);
  top: 50%;
  left: 50%;
`;

const Canvas = styled.canvas`
  position: absolute;
  transform: translate(-50%, -50%);
  top: 50%;
  left: 50%;
`;

const Text = styled.div``;
