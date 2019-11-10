# CS7641-term-project
Using reinforcement learning to solve Settlers of Catan

## JSettlers:

  - Newest JAR file available for download at http://nand.net/jsettlers/ doesn't have newest features. Instead, you must download the source repository over at https://github.com/jdmonin/JSettlers2 and build the Gradle project into JAR files yourself. The two built files for server and client are in the root directory under JSettlers-2.0.000.jar (client) and JSettlersServer-2.0.00.jar (server).
  
  - The JARs available have been built from the modified JSettlers project included in this repository. 
  
## Developing an AI Agent:
  - To develop an AI agent, we'll need to utilize some commands provided by the newest version of JSettlers. These are creating our own bots, connecting these bots to a game server, and playing games with them. 

### Creating a bot
     - To define your own bot, you need to use the files in the directory JSettlers2-master/src/main/java/soc/robot/sample3p as an example
     - You have to make two classes: AgentBrain and AgentClient, inherited from SOCRobotBrain and SOCRobotClient respectively. The client handles server-client communications, while the brain handles the decision logic. 
     - Once you have these classes, you need to create a package with them (ex. soc.robot.sample3p)
     
### Connecting a bot to a game server
     - Once your bot has been created (and you've rebuilt the JAR files with the new code), you can connect to a started server by running the following commands:
       - [Server] : java -jar JSettlersServer-2.0.00.jar -Djsettlers.bots.cookie=foo
         - The cookie allows robots to connect to the server and be added to the list of bots, we can define it to be whatever we want
       - [Client] : java -cp JSettlers-2.0.00.jar soc.robot.sample3p.Sample3PClient localhost 8880 robot1 rb1 foo
         - The command follows the structure 'server port username id cookie', and the cookie must match the server for the robot to be accepted as a connection
         - The client object should be whatever new bot you created earlier
         
### Playing games with a bot
     - It seems like the server always has a list of bot objects, and whenever the server wants to build a game with bots it will randomly sample from this list to add agents as AI. We want to control this sampling to deterministically decide which agents are playing every game (whether it's JSettlers default agent or a custom agent). Suppose the scenario where we want to play one custom AI against three JSettlers' AI:
        - We can utilize very useful command-line arguments to set up this scenario
        - [Server] : java -jar JSettlersServer-2.0.00.jar -Djsettlers.bots.cookie=foo -Djsettlers.bots.startbots=3 -Djsettlers.jsettlers.bots.percent3p=25
        - This will instantiate the default list of bots to be 4 agents, 25% of which are our custom agent.
     - After doing so, we want to have them play a game. We can do also have this occur right when we start the server with another command-line flag: 
        - [Server] : java -jar JSettlersServer-2.0.00.jar -Djsettlers.bots.cookie=foo -Djsettlers.bots.startbots=3 -Djsettlers.jsettlers.bots.percent3p=25 -Djsettlers.bots.botgames.total=5
        - This will automatically create and start 5 games, where each game starts when the previous game finishes. You can connect to the server as a regular client and observe the game if you want.
        
        
        
## Automating games with bots
     - As of Oct 20th, the server now automatically spins up several instances of our custom agent when the server starts, meaning that manual connections aren't necessary.
        - The following command will automatically play 10 games in a row, each game comprising of 3 JSettlers agents and 1 Custom Agent
        - [Server] : java -jar JSettlersServer-2.0.00.jar -Djsettlers.jsettlers.bots.percent3p=25 -Djsettlers.bots.botgames.total=10
        - No client connections are necessary with this call; but you can connect a human-player client to join and spectate the games as they take place.
        
        
### Promising Approach:

    - Create a python server that's running the DQN agent. In Jsettlers, write code for a custom SOCRobotBrain that will connect to our DQN server when instantiated, and will send over the game state whenever it is the agent's turn. The python server can receive this data, process it with the neural network, and send back the best action to take. This requires learning the object details of a SOCGameState, but it seems like it's a proper way to connect a DQN agent to this Java Framework.
    - Overall, the structure will look like:
       - Instantiate JSettlers server
       - Instantiate DQN server
       - Initialize our agent's Client to connect to the JSettlers server
       - Initialize our agent's Brain to connect to the DQN server
       - The JSettlers server will query our Client, which will query our Brain, which will query our DQN server, which will return a decision to our Brain, which will return to our Client, which will return to the JSettlers server.
          - All the communication is already built-in except for querying and returning from our DQN Server.
          
          
## Usage

Instantiate JSettlers server:

```
java -jar JSettlersServer-2.0.00.jar -Djsettlers.bots.percent3p=25 -Djsettlers.bots.botgames.total=50
```

Instantiate DQN server:

```
python dqn_server.py
```

Run Tensorboard:

```
tensorboard --logdir=logs/
```
