document.addEventListener("DOMContentLoaded", function () {
  const aboutMeButton = document.getElementsByClassName("aboutMeButton");
  const blogButton = document.getElementsByClassName("blogButton");
  const experienceButton = document.getElementsByClassName("experienceButton");
  const projectButton = document.getElementsByClassName("projectButton");
  const contactButton = document.getElementsByClassName("contactButton");
  const languageButton = document.getElementsByClassName("languageButton");

  console.log(aboutMeButton);
  console.log(blogButton);
  console.log(experienceButton);
  console.log(projectButton);
  console.log(contactButton);
  console.log(languageButton);

  Array.from(aboutMeButton).forEach(function (button) {
    button.addEventListener("click", function () {
      console.log("aboutMeButton button clicked!");
      window.location.href = "/";
    });
    button.addEventListener("mousedown", function () {
      button.style.color = "rgb(0, 255, 255)";
    });
  });

  Array.from(blogButton).forEach(function (button) {
    button.addEventListener("click", function () {
      console.log("Blog button clicked!");
      window.location.href = "/blog";
    });
    button.addEventListener("mousedown", function () {
      button.style.color = "rgb(0, 255, 255)";
    });
  });

  Array.from(experienceButton).forEach(function (button) {
    button.addEventListener("click", function () {
      console.log("experienceButton button clicked!");
      window.location.href = "/experience";
    });
    button.addEventListener("mousedown", function () {
      button.style.color = "rgb(0, 255, 255)";
    });
  });

  Array.from(projectButton).forEach(function (button) {
    button.addEventListener("click", function () {
      console.log("projectButton button clicked!");
      window.location.href = "/project";
    });
    button.addEventListener("mousedown", function () {
      button.style.color = "rgb(0, 255, 255)";
    });
  });

  Array.from(contactButton).forEach(function (button) {
    button.addEventListener("click", function () {
      console.log("contactButton button clicked!");
      window.location.href = "/contact";
    });
    button.addEventListener("mousedown", function () {
      button.style.color = "rgb(0, 255, 255)";
    });
  });
});

function toggleFolder(element) {
  let nestedList = element.nextElementSibling;
  if (nestedList) {
    nestedList.classList.toggle("active");
  }
}

const images = ["static/images/defaultPic1.png", "static/images/myProfile2.jpg", "static/images/myProfile.jpg",];
let index = 0;
  
document.querySelector(".profile").addEventListener("click", function () {
  index = (index + 1) % images.length;
  this.style.backgroundImage = `url("${images[index]}")`;
});

function openModalViews() {
  document.querySelector(".modal.views").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openModalUrls() {
  document.querySelector(".modal.urls").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openModalSty() {
  document.querySelector(".modal.sty").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openModalScr() {
  document.querySelector(".modal.scr").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openModalBa() {
  document.querySelector(".modal.ba").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openModalThisWeb() {
  document.querySelector(".modal.thisWeb").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openDCBotPic() {
  document.querySelector(".modal.dcBotPic").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openDCBotCode() {
  document.querySelector(".modal.dcBotCode").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openSokobanCode() {
  document.querySelector(".modal.sokobanCode").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openSokobanPic() {
  document.querySelector(".modal.sokobanPic").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openWinSokobanPic() {
  document.querySelector(".modal.winSokobanPic").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openOthello() {
  document.querySelector(".modal.othello").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openOthelloRR() {
  document.querySelector(".modal.othelloRR").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openOthelloHR() {
  document.querySelector(".modal.othelloHR").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWAITest() {
  document.querySelector(".modal.UWAITest").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWCardPileTest() {
  document.querySelector(".modal.UWCardPileTest").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWCardTest() {
  document.querySelector(".modal.UWCardTest").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWDeckTest() {
  document.querySelector(".modal.UWDeckTest").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWHandTest() {
  document.querySelector(".modal.UWHandTest").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWAI() {
  document.querySelector(".modal.UWAI").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWBiggestCardAI() {
  document.querySelector(".modal.UWBiggestCardAI").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWCard() {
  document.querySelector(".modal.UWCard").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWCardPile() {
  document.querySelector(".modal.UWCardPile").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWDeck() {
  document.querySelector(".modal.UWDeck").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWHand() {
  document.querySelector(".modal.UWHand").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWSmallestCardAI() {
  document.querySelector(".modal.UWSmallestCardAI").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWTournament() {
  document.querySelector(".modal.UWTournament").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openUWUnoWarMatch() {
  document.querySelector(".modal.UWUnoWarMatch").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function openVector3() {
  document.querySelector(".modal.vector3").style.display = "block";
  document.querySelector(".modal-bg").style.display = "block";
}

function closeModal() {
  document.querySelectorAll(".modal").forEach(modal => {
    modal.style.display = "none";
  });
  document.querySelector(".modal-bg").style.display = "none";
}